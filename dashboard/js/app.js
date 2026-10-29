// Command Center bootstrap: wires the API, live socket, store and renderers together.

import { api, LiveSocket } from "./api.js";
import { AudioAlerts } from "./audio.js";
import { HEALTH_POLL_MS, RELATIVE_TIME_REFRESH_MS, SAMPLE_CALLS, SHORTCUTS, STATUS_LABELS } from "./config.js";
import {
    state, subscribe, notify, setState, setFilter, clearFilters, upsertIncident, replaceIncidents,
    selectIncident, selectedIncident, sortedIncidents, visibleIncidents, hasActiveFilters, computeStats,
} from "./store.js";
import {
    el, renderHealth, renderStats, renderDepartments, renderVolume, renderShortcuts,
    renderFeed, renderDetail, toast,
} from "./render.js";

const $ = (id) => document.getElementById(id);
const audio = new AudioAlerts();
const BASE_TITLE = "NEXUS Dispatch";

// ---------- Rendering (coalesced to one paint per frame) ----------

let renderQueued = false;
function scheduleRender() {
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(() => {
        renderQueued = false;
        render();
    });
}

function render() {
    const now = Date.now();
    const stats = computeStats(now);
    const visible = visibleIncidents();

    renderHealth(state.health, state.socket);
    renderStats(stats);
    renderDepartments(stats, state.config.departments, state.filters.department);
    renderVolume(stats);
    renderFeed({
        incidents: visible,
        total: state.incidents.size,
        selectedId: state.selectedId,
        freshIds: state.freshIds,
        hasFilters: hasActiveFilters(),
        threshold: state.config.confidence_threshold,
        now,
    });
    renderDetail(selectedIncident(), state.config, now);

    document.title = stats.unacknowledgedCritical
        ? `(${stats.unacknowledgedCritical}) ⚠ ${BASE_TITLE}`
        : BASE_TITLE;
}

subscribe(scheduleRender);

// ---------- Data loading ----------

async function refreshHealth() {
    try {
        setState({ health: await api.health() }, "health");
    } catch {
        setState({ health: null }, "health");
    }
}

async function resyncIncidents() {
    try {
        replaceIncidents(await api.incidents(500));
    } catch (error) {
        toast("Could not load incident history", error.message, "warn");
    }
}

async function loadConfig() {
    try {
        state.config = { ...state.config, ...(await api.config()) };
    } catch {
        // Fall back to DEFAULT_CONFIG; the health banner reports the outage
    }
    const select = $("filter-department");
    select.replaceChildren(
        el("option", { value: "", text: "All departments" }),
        ...state.config.departments.map((d) => el("option", { value: d, text: d })),
    );
}

// ---------- Live socket ----------

let hasConnectedBefore = false;

const socket = new LiveSocket({
    onStatus(status) {
        setState({ socket: status }, "socket");
        if (status === "open") {
            if (hasConnectedBefore) {
                toast("Live feed restored", "Resynchronising missed calls…", "ok", 3000);
                resyncIncidents();
                refreshHealth();
            }
            hasConnectedBefore = true;
        }
    },
    onMessage(message) {
        if (!message.incident) return;
        if (message.type === "incident_created") {
            const isNew = upsertIncident(message.incident, { fresh: true });
            if (isNew) announce(message.incident);
        } else if (message.type === "incident_updated") {
            upsertIncident(message.incident);
        }
        notify("incidents");
    },
});

function announce(incident) {
    audio.play(incident);
    if (incident.severity_flag) {
        toast(`High severity · ${incident.assigned_department}`, incident.transcript, "danger", 7000);
    } else if (incident.needs_human_review) {
        toast("Low-confidence call needs review",
            `${incident.id}: model is ${Math.round(incident.confidence * 100)}% sure of ${incident.assigned_department}`, "warn");
    }
    $("live-announcer").textContent =
        `New ${incident.severity_flag ? "high severity " : ""}${incident.assigned_department} call from ${incident.caller_id}`;
}

// ---------- Dispatcher actions ----------

async function updateIncident(id, patch, successMessage) {
    try {
        upsertIncident(await api.updateIncident(id, patch));
        notify("incidents");
        if (successMessage) toast(successMessage, id, "ok", 2500);
        return true;
    } catch (error) {
        toast("Update failed", error.message, "danger");
        return false;
    }
}

function setStatus(status) {
    const incident = selectedIncident();
    if (!incident || incident.status === status) return;
    updateIncident(incident.id, { status }, `Marked ${STATUS_LABELS[status].toLowerCase()}`);
}

function reroute() {
    const incident = selectedIncident();
    const department = $("reroute-select").value;
    if (!incident || department === incident.assigned_department) return;
    updateIncident(incident.id, { assigned_department: department }, `Rerouted to ${department}`);
}

async function addNote() {
    const incident = selectedIncident();
    const input = $("note-input");
    const text = input.value.trim();
    if (!incident || !text) return;
    if (await updateIncident(incident.id, { note: text }, "Note added")) {
        const fresh = $("note-input");
        if (fresh) fresh.value = "";
    }
}

$("detail").addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button || button.disabled) return;
    const { action, status } = button.dataset;
    if (action === "status") setStatus(status);
    else if (action === "reroute") reroute();
    else if (action === "note") addNote();
    else if (action === "close") selectIncident(null);
});

$("detail").addEventListener("keydown", (event) => {
    if (event.target.id === "note-input" && event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        addNote();
    }
});

// ---------- Feed & filters ----------

function select(id, { scroll = false } = {}) {
    selectIncident(id);
    if (scroll) {
        requestAnimationFrame(() => {
            const row = document.querySelector(`.incident[data-id="${CSS.escape(id)}"]`);
            if (row) row.scrollIntoView({ block: "nearest" });
        });
    }
}

$("feed-list").addEventListener("click", (event) => {
    const row = event.target.closest(".incident");
    if (row) select(row.dataset.id);
});

$("dept-bars").addEventListener("click", (event) => {
    const row = event.target.closest(".dept-row");
    if (!row) return;
    const value = state.filters.department === row.dataset.dept ? "" : row.dataset.dept;
    $("filter-department").value = value;
    setFilter("department", value);
});

let searchTimer;
$("filter-search").addEventListener("input", (event) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => setFilter("search", event.target.value), 120);
});

for (const [id, key] of [["filter-department", "department"], ["filter-severity", "severity"], ["filter-status", "status"]]) {
    $(id).addEventListener("change", (event) => setFilter(key, event.target.value));
}

$("filter-review").addEventListener("click", (event) => {
    const pressed = event.currentTarget.getAttribute("aria-pressed") !== "true";
    event.currentTarget.setAttribute("aria-pressed", String(pressed));
    setFilter("reviewOnly", pressed);
});

function resetFilters() {
    $("filter-search").value = "";
    $("filter-department").value = "";
    $("filter-severity").value = "";
    $("filter-status").value = "";
    $("filter-review").setAttribute("aria-pressed", "false");
    clearFilters();
}

$("btn-clear-filters").addEventListener("click", resetFilters);

// ---------- Audio ----------

function toggleAudio() {
    const on = audio.toggle();
    const button = $("btn-audio");
    button.setAttribute("aria-pressed", String(on));
    button.querySelector(".btn-label").textContent = on ? "Audio on" : "Audio off";
    toast(on ? "Audio alerts enabled" : "Audio alerts muted", "", "info", 2000);
}

$("btn-audio").addEventListener("click", toggleAudio);

// ---------- CSV export ----------

function csvCell(value) {
    let text = value === null || value === undefined ? "" : String(value);
    if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`; // block spreadsheet formula injection
    return `"${text.replace(/"/g, '""')}"`;
}

function exportCsv() {
    const rows = visibleIncidents();
    if (!rows.length) {
        toast("Nothing to export", "The current view has no calls.", "warn");
        return;
    }
    const columns = ["id", "received_at", "caller_id", "assigned_department", "model_department", "severity_flag",
        "confidence", "needs_human_review", "status", "overridden", "transcript"];
    const lines = [columns.join(","), ...rows.map((r) => columns.map((c) => csvCell(r[c])).join(","))];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = el("a", { href: url, download: `dispatch-incidents-${new Date().toISOString().slice(0, 19)}.csv` });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("Export ready", `${rows.length} calls exported`, "ok", 2500);
}

$("btn-export").addEventListener("click", exportCsv);

// ---------- New call dialog ----------

const dialog = $("new-call-dialog");
const form = $("new-call-form");

function randomCallerId() {
    const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    return `MAN-${Array.from({ length: 6 }, () => chars[Math.floor(Math.random() * chars.length)]).join("")}`;
}

function openNewCall() {
    form.reset();
    $("new-call-caller").value = randomCallerId();
    $("new-call-error").textContent = "";
    dialog.showModal();
    $("new-call-transcript").focus();
}

$("btn-new-call").addEventListener("click", openNewCall);
$("new-call-cancel").addEventListener("click", () => dialog.close());
$("new-call-close").addEventListener("click", () => dialog.close());

$("new-call-samples").replaceChildren(...SAMPLE_CALLS.map((text, i) => el("button", {
    class: "btn", type: "button", title: text, dataset: { sample: String(i) },
}, `Sample ${i + 1}`)));

$("new-call-samples").addEventListener("click", (event) => {
    const button = event.target.closest("[data-sample]");
    if (button) $("new-call-transcript").value = SAMPLE_CALLS[Number(button.dataset.sample)];
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = $("new-call-submit");
    const callerId = $("new-call-caller").value.trim();
    const transcript = $("new-call-transcript").value.trim();
    if (!callerId || !transcript) {
        $("new-call-error").textContent = "Caller ID and transcript are both required.";
        return;
    }

    submit.disabled = true;
    submit.textContent = "Triaging…";
    try {
        const incident = await api.triage(callerId, transcript);
        if (upsertIncident(incident, { fresh: true })) announce(incident);
        dialog.close();
        select(incident.id, { scroll: true });
    } catch (error) {
        $("new-call-error").textContent = error.status === 503
            ? "The triage model is not loaded. Check the API server."
            : error.message;
    } finally {
        submit.disabled = false;
        submit.textContent = "Triage call";
    }
});

// ---------- Keyboard shortcuts ----------

function moveSelection(step) {
    const list = visibleIncidents();
    if (!list.length) return;
    const index = list.findIndex((i) => i.id === state.selectedId);
    const next = index === -1 ? 0 : Math.min(list.length - 1, Math.max(0, index + step));
    select(list[next].id, { scroll: true });
}

document.addEventListener("keydown", (event) => {
    if (dialog.open || event.metaKey || event.ctrlKey || event.altKey) return;
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName);

    if (event.key === "Escape") {
        if (typing) event.target.blur();
        else if (state.selectedId) selectIncident(null);
        else if (hasActiveFilters()) resetFilters();
        return;
    }
    if (typing) return;

    const actions = {
        "/": () => $("filter-search").focus(),
        j: () => moveSelection(1),
        k: () => moveSelection(-1),
        a: () => setStatus("ACKNOWLEDGED"),
        d: () => setStatus("DISPATCHED"),
        r: () => setStatus("RESOLVED"),
        n: openNewCall,
        m: toggleAudio,
    };
    const action = actions[event.key];
    if (action) {
        event.preventDefault();
        action();
    }
});

// ---------- Clock & timers ----------

function tickClock() {
    $("clock").textContent = new Date().toLocaleTimeString([], { hour12: false });
}

// ---------- Boot ----------

async function boot() {
    renderShortcuts(SHORTCUTS);
    tickClock();
    setInterval(tickClock, 1000);
    setInterval(refreshHealth, HEALTH_POLL_MS);
    setInterval(scheduleRender, RELATIVE_TIME_REFRESH_MS);

    await loadConfig();
    await Promise.all([refreshHealth(), resyncIncidents()]);
    socket.connect();

    const newest = sortedIncidents()[0];
    if (newest && window.matchMedia("(min-width: 1281px)").matches) selectIncident(newest.id);
    scheduleRender();
}

boot();
