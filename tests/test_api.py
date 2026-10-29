import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from api import main
from core.architecture import GPTModel
from core.config import NUM_CLASSES, ROUTING_MAP, HIGH_SEVERITY_CLASSES

TINY_CONFIG = {"vocab_size": 50257, "context_length": 64, "emb_dim": 16, "n_heads": 2,
               "n_layers": 1, "drop_rate": 0.0, "qkv_bias": True}

CALL = {"caller_id": "TEST-001", "transcript": "Emergency on Jasper Ave, someone is engulfed in flames."}


def install_tiny_model(forced_class=None):
    """Swap in a tiny model whose head always predicts `forced_class` (or is uniform if None)."""
    model = GPTModel(TINY_CONFIG)
    model.out_head = nn.Linear(TINY_CONFIG["emb_dim"], NUM_CLASSES)
    with torch.no_grad():
        model.out_head.weight.zero_()
        model.out_head.bias.zero_()
        if forced_class is not None:
            model.out_head.bias[forced_class] = 10.0
    main.state["device"] = torch.device("cpu")
    main.state["model"] = model.eval()


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Point startup at a missing file so tests never load the real 548MB artifact
    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(tmp_path / "missing.pth"))
    main.incidents.clear()
    with TestClient(main.app) as c:
        yield c


def test_returns_503_when_model_not_loaded(client):
    response = client.post("/v1/triage/analyze", json=CALL)
    assert response.status_code == 503


@pytest.mark.parametrize("class_idx", range(NUM_CLASSES))
def test_routing_and_severity_per_class(client, class_idx):
    install_tiny_model(forced_class=class_idx)
    response = client.post("/v1/triage/analyze", json=CALL)
    assert response.status_code == 200
    body = response.json()
    assert body["assigned_department"] == ROUTING_MAP[class_idx]
    assert body["severity_flag"] == (class_idx in HIGH_SEVERITY_CLASSES)
    assert body["confidence"] > 0.99
    assert body["needs_human_review"] is False
    assert body["caller_id"] == CALL["caller_id"]
    assert body["transcript"] == CALL["transcript"]


def test_low_confidence_is_flagged_for_human_review(client):
    install_tiny_model(forced_class=None)  # uniform: 1/5 = 0.2 confidence
    body = client.post("/v1/triage/analyze", json=CALL).json()
    assert body["confidence"] == pytest.approx(1 / NUM_CLASSES, abs=1e-3)
    assert body["needs_human_review"] is True


@pytest.mark.parametrize("payload", [
    {"caller_id": "X"},
    {"transcript": "help"},
    {"caller_id": "X", "transcript": ""},
])
def test_invalid_payload_returns_422(client, payload):
    install_tiny_model(forced_class=0)
    assert client.post("/v1/triage/analyze", json=payload).status_code == 422


def test_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "NEXUS" in response.text


def test_triage_result_is_broadcast_to_dashboard(client):
    install_tiny_model(forced_class=1)
    with client.websocket_connect("/ws/dashboard") as ws:
        client.post("/v1/triage/analyze", json=CALL)
        message = ws.receive_json()
    assert message["type"] == "incident_created"
    assert message["incident"]["assigned_department"] == "FIRE"
    assert message["incident"]["severity_flag"] is True


def test_websocket_heartbeat(client):
    with client.websocket_connect("/ws/dashboard") as ws:
        ws.send_text("ping")
        assert ws.receive_json() == {"type": "pong"}


def test_triage_creates_incident_with_probabilities(client):
    install_tiny_model(forced_class=0)
    body = client.post("/v1/triage/analyze", json=CALL).json()
    assert body["id"].startswith("INC-")
    assert body["status"] == "NEW"
    assert body["model_department"] == "MEDICAL"
    assert set(body["probabilities"]) == set(ROUTING_MAP.values())
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_incidents_are_listed_newest_first(client):
    install_tiny_model(forced_class=0)
    first = client.post("/v1/triage/analyze", json=CALL).json()
    second = client.post("/v1/triage/analyze", json=CALL).json()
    listed = client.get("/v1/incidents").json()
    assert [i["id"] for i in listed] == [second["id"], first["id"]]
    assert client.get(f"/v1/incidents/{first['id']}").json()["id"] == first["id"]
    assert client.get("/v1/incidents/INC-MISSING").status_code == 404


def test_status_update_is_recorded_and_broadcast(client):
    install_tiny_model(forced_class=0)
    incident = client.post("/v1/triage/analyze", json=CALL).json()
    with client.websocket_connect("/ws/dashboard") as ws:
        updated = client.patch(f"/v1/incidents/{incident['id']}",
                               json={"status": "ACKNOWLEDGED", "note": "Unit 12 en route"}).json()
        message = ws.receive_json()
    assert updated["status"] == "ACKNOWLEDGED"
    assert updated["notes"][0]["text"] == "Unit 12 en route"
    assert [h["action"] for h in updated["history"]] == ["CREATED", "ACKNOWLEDGED", "NOTE"]
    assert message == {"type": "incident_updated", "incident": updated}


def test_department_override_recomputes_severity(client):
    install_tiny_model(forced_class=3)  # NON-VIOLENT CRIME, standard severity
    incident = client.post("/v1/triage/analyze", json=CALL).json()
    updated = client.patch(f"/v1/incidents/{incident['id']}",
                           json={"assigned_department": "VIOLENT CRIME"}).json()
    assert updated["assigned_department"] == "VIOLENT CRIME"
    assert updated["model_department"] == "NON-VIOLENT CRIME"
    assert updated["severity_flag"] is True
    assert updated["overridden"] is True


@pytest.mark.parametrize("payload, code", [
    ({"status": "ON_HOLD"}, 422),
    ({"assigned_department": "SPACE FORCE"}, 422),
])
def test_invalid_incident_update(client, payload, code):
    install_tiny_model(forced_class=0)
    incident = client.post("/v1/triage/analyze", json=CALL).json()
    assert client.patch(f"/v1/incidents/{incident['id']}", json=payload).status_code == code


def test_update_unknown_incident_returns_404(client):
    assert client.patch("/v1/incidents/INC-MISSING", json={"status": "RESOLVED"}).status_code == 404


def test_health_reports_model_state(client):
    assert client.get("/v1/health").json()["model_loaded"] is False
    install_tiny_model(forced_class=0)
    health = client.get("/v1/health").json()
    assert health["model_loaded"] is True
    assert health["status"] == "ok"


def test_config_exposes_routing_rules(client):
    config = client.get("/v1/config").json()
    assert config["departments"] == [ROUTING_MAP[i] for i in range(NUM_CLASSES)]
    assert config["statuses"] == ["NEW", "ACKNOWLEDGED", "DISPATCHED", "RESOLVED"]
    assert "MEDICAL" not in config["high_severity_departments"]
