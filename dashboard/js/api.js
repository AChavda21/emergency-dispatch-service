// Network layer: REST calls to the triage API and the live dashboard socket.

async function request(path, options = {}) {
    const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });
    let body = null;
    try { body = await response.json(); } catch { /* empty or non-JSON body */ }

    if (!response.ok) {
        const detail = body && body.detail;
        const message = Array.isArray(detail)
            ? detail.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join("; ")
            : detail || `${response.status} ${response.statusText}`;
        const error = new Error(message);
        error.status = response.status;
        throw error;
    }
    return body;
}

export const api = {
    health: () => request("/v1/health"),
    config: () => request("/v1/config"),
    incidents: (limit = 200) => request(`/v1/incidents?limit=${limit}`),
    triage: (callerId, transcript) => request("/v1/triage/analyze", {
        method: "POST",
        body: JSON.stringify({ caller_id: callerId, transcript }),
    }),
    updateIncident: (id, patch) => request(`/v1/incidents/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(patch),
    }),
};

const HEARTBEAT_MS = 20000;
const HEARTBEAT_TIMEOUT_MS = 10000;
const MAX_BACKOFF_MS = 30000;

// WebSocket client with exponential-backoff reconnect and a ping/pong heartbeat
// so half-open connections (laptop sleep, proxy timeouts) are detected and replaced.
export class LiveSocket {
    constructor({ path = "/ws/dashboard", onMessage, onStatus }) {
        this.path = path;
        this.onMessage = onMessage;
        this.onStatus = onStatus;
        this.attempt = 0;
        this.ws = null;
        this.heartbeatTimer = null;
        this.pongTimer = null;
        this.reconnectTimer = null;
    }

    get url() {
        const scheme = window.location.protocol === "https:" ? "wss" : "ws";
        return `${scheme}://${window.location.host}${this.path}`;
    }

    connect() {
        clearTimeout(this.reconnectTimer);
        this.onStatus("connecting", this.attempt);
        const ws = new WebSocket(this.url);
        this.ws = ws;

        ws.addEventListener("open", () => {
            this.attempt = 0;
            this.onStatus("open");
            this.startHeartbeat();
        });

        ws.addEventListener("message", (event) => {
            let message;
            try { message = JSON.parse(event.data); } catch { return; }
            if (message.type === "pong") {
                clearTimeout(this.pongTimer);
                return;
            }
            this.onMessage(message);
        });

        ws.addEventListener("close", () => {
            if (this.ws !== ws) return;
            this.stopHeartbeat();
            this.scheduleReconnect();
        });

        ws.addEventListener("error", () => ws.close());
    }

    scheduleReconnect() {
        const base = Math.min(MAX_BACKOFF_MS, 1000 * 2 ** this.attempt);
        const delay = base / 2 + Math.random() * (base / 2);
        this.attempt += 1;
        this.onStatus("closed", this.attempt, delay);
        this.reconnectTimer = setTimeout(() => this.connect(), delay);
    }

    startHeartbeat() {
        this.stopHeartbeat();
        this.heartbeatTimer = setInterval(() => {
            if (this.ws.readyState !== WebSocket.OPEN) return;
            this.ws.send("ping");
            clearTimeout(this.pongTimer);
            this.pongTimer = setTimeout(() => this.ws.close(), HEARTBEAT_TIMEOUT_MS);
        }, HEARTBEAT_MS);
    }

    stopHeartbeat() {
        clearInterval(this.heartbeatTimer);
        clearTimeout(this.pongTimer);
    }

    reconnectNow() {
        this.attempt = 0;
        if (this.ws) {
            const old = this.ws;
            this.ws = null;
            this.stopHeartbeat();
            old.close();
        }
        this.connect();
    }
}
