# The Production Web Server
# This is persistent microservice.
# It uses FastAPI to load the model into memory at startup, then waits to process JSON network requests in milliseconds.

import sys
import os
import time
import torch
import torch.nn as nn
import tiktoken
import logging
import json
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal

# Append project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.architecture import GPTModel
from core.config import (MODEL_CONFIG, NUM_CLASSES, ROUTING_MAP, HIGH_SEVERITY_CLASSES,
                         CONFIDENCE_THRESHOLD, MODEL_PATH, DASHBOARD_PATH, select_device)
from core.preprocessing import to_tensors, select_last_token
from api.incidents import IncidentStore, STATUSES

API_VERSION = "3.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TriageAPI")


# --- WEBSOCKET MANAGER FOR LIVE DASHBOARD ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Iterate over a copy so a dead socket can be dropped mid-broadcast
        for connection in list(self.active_connections):
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                logger.warning("Dropping dashboard connection that failed to receive a message.")
                self.disconnect(connection)

manager = ConnectionManager()
incidents = IncidentStore()

# --- ML MODEL SETUP ---
state = {"model": None, "tokenizer": None, "device": None, "model_path": None, "started_at": None}

class CallTranscript(BaseModel):
    caller_id: str = Field(min_length=1, max_length=64)
    transcript: str = Field(min_length=1, max_length=2000)

class IncidentUpdate(BaseModel):
    status: Optional[Literal[STATUSES]] = None
    assigned_department: Optional[str] = None
    note: Optional[str] = Field(default=None, max_length=500)


def load_model(path, device):
    model = GPTModel(MODEL_CONFIG)
    model.out_head = nn.Linear(MODEL_CONFIG["emb_dim"], NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval().to(device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["started_at"] = time.time()
    state["device"] = select_device()
    state["tokenizer"] = tiktoken.get_encoding("gpt2")
    state["model_path"] = os.environ.get("TRIAGE_MODEL_PATH", str(MODEL_PATH))

    try:
        state["model"] = load_model(state["model_path"], state["device"])
        logger.info(f"Model {state['model_path']} mounted on {state['device']} and ready for inference.")
    except Exception as e:
        logger.error(f"Failed to load artifact {state['model_path']}: {str(e)}")
    yield
    state["model"] = None


app = FastAPI(title="Emergency Dispatch Triage API", version=API_VERSION, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=DASHBOARD_PATH.parent), name="static")


# --- API ENDPOINTS ---
@app.post("/v1/triage/analyze")
async def analyze_transcript(call: CallTranscript):
    model = state["model"]
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_ids, lengths = to_tensors(state["tokenizer"], [call.transcript], state["device"])

    with torch.no_grad():
        logits = select_last_token(model(input_ids), lengths)
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted_idx = probabilities.max(dim=-1)
        confidence, predicted_idx = confidence.item(), predicted_idx.item()

    department = ROUTING_MAP[predicted_idx]
    severity = predicted_idx in HIGH_SEVERITY_CLASSES

    # 1. Record the incident
    incident = incidents.create({
        "caller_id": call.caller_id,
        "assigned_department": department,
        "severity_flag": severity,
        "confidence": round(confidence, 4),
        "needs_human_review": confidence < CONFIDENCE_THRESHOLD,
        "probabilities": {ROUTING_MAP[i]: round(p, 4) for i, p in enumerate(probabilities[0].tolist())},
        "transcript": call.transcript
    })

    # 2. Push to the live dashboard instantly
    await manager.broadcast({"type": "incident_created", "incident": incident})

    return incident

@app.get("/v1/incidents")
async def list_incidents(limit: int = Query(200, ge=1, le=500)):
    return incidents.recent(limit)

@app.get("/v1/incidents/{incident_id}")
async def get_incident(incident_id: str):
    incident = incidents.get(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident

@app.patch("/v1/incidents/{incident_id}")
async def update_incident(incident_id: str, update: IncidentUpdate):
    try:
        incident = incidents.update(incident_id, status=update.status,
                                    department=update.assigned_department, note=update.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="Incident not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    await manager.broadcast({"type": "incident_updated", "incident": incident})
    return incident

@app.get("/v1/health")
async def health():
    return {
        "status": "ok" if state["model"] else "degraded",
        "model_loaded": state["model"] is not None,
        "model_path": os.path.basename(state["model_path"] or ""),
        "device": str(state["device"]),
        "uptime_seconds": round(time.time() - state["started_at"]) if state["started_at"] else 0,
        "connected_dashboards": len(manager.active_connections),
        "version": API_VERSION,
    }

@app.get("/v1/config")
async def dashboard_config():
    return {
        "departments": [ROUTING_MAP[i] for i in range(NUM_CLASSES)],
        "high_severity_departments": [ROUTING_MAP[i] for i in sorted(HIGH_SEVERITY_CLASSES)],
        "statuses": list(STATUSES),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            if await websocket.receive_text() == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
async def get_dashboard():
    with open(DASHBOARD_PATH, "r") as f:
        return HTMLResponse(f.read())
