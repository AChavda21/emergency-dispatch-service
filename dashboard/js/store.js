// Client-side state. Incidents are keyed by id so live events, resyncs after
// reconnect and local actions all merge into one record without duplicates.

import { DEFAULT_CONFIG, MAX_CLIENT_INCIDENTS, VOLUME_WINDOW_MINUTES, RATE_WINDOW_MINUTES } from "./config.js";

const listeners = new Set();

export const state = {
    config: DEFAULT_CONFIG,
    health: null,
    socket: "connecting",
    incidents: new Map(),
    selectedId: null,
    freshIds: new Set(),
    filters: { search: "", department: "", severity: "", status: "", reviewOnly: false },
};

export function subscribe(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

export function notify(reason) {
    listeners.forEach((listener) => listener(reason));
}

export function setState(patch, reason = "state") {
    Object.assign(state, patch);
    notify(reason);
}

export function setFilter(key, value) {
    state.filters = { ...state.filters, [key]: value };
    notify("filters");
}

export function clearFilters() {
    state.filters = { search: "", department: "", severity: "", status: "", reviewOnly: false };
    notify("filters");
}

/** Insert or replace an incident. Returns true if it was not known before. */
export function upsertIncident(incident, { fresh = false } = {}) {
    const isNew = !state.incidents.has(incident.id);
    state.incidents.set(incident.id, incident);
    if (fresh && isNew) state.freshIds.add(incident.id);

    if (state.incidents.size > MAX_CLIENT_INCIDENTS) {
        const oldest = sortedIncidents().slice(MAX_CLIENT_INCIDENTS);
        oldest.forEach((i) => { if (i.id !== state.selectedId) state.incidents.delete(i.id); });
    }
    return isNew;
}

export function replaceIncidents(list) {
    // The server list is authoritative, but keep anything that arrived over the socket meanwhile
    list.forEach((incident) => {
        const local = state.incidents.get(incident.id);
        if (!local || local.updated_at <= incident.updated_at) state.incidents.set(incident.id, incident);
    });
    notify("incidents");
}

export function selectIncident(id) {
    state.selectedId = id;
    notify("selection");
}

export function selectedIncident() {
    return state.selectedId ? state.incidents.get(state.selectedId) || null : null;
}

export function sortedIncidents() {
    return [...state.incidents.values()].sort((a, b) => (a.received_at < b.received_at ? 1 : -1));
}

export function visibleIncidents() {
    const { search, department, severity, status, reviewOnly } = state.filters;
    const needle = search.trim().toLowerCase();
    return sortedIncidents().filter((i) => {
        if (department && i.assigned_department !== department) return false;
        if (severity === "high" && !i.severity_flag) return false;
        if (severity === "standard" && i.severity_flag) return false;
        if (status === "OPEN" && i.status === "RESOLVED") return false;
        if (status && status !== "OPEN" && i.status !== status) return false;
        if (reviewOnly && !i.needs_human_review) return false;
        if (needle) {
            const haystack = `${i.id} ${i.caller_id} ${i.transcript} ${i.assigned_department}`.toLowerCase();
            if (!haystack.includes(needle)) return false;
        }
        return true;
    });
}

export function hasActiveFilters() {
    const f = state.filters;
    return Boolean(f.search || f.department || f.severity || f.status || f.reviewOnly);
}

export function computeStats(now = Date.now()) {
    const all = [...state.incidents.values()];
    const byDepartment = Object.fromEntries(state.config.departments.map((d) => [d, 0]));
    const volume = Array.from({ length: VOLUME_WINDOW_MINUTES }, () => ({ total: 0, critical: 0 }));
    let critical = 0, review = 0, open = 0, unacknowledgedCritical = 0, confidenceSum = 0, recent = 0;

    for (const i of all) {
        if (i.severity_flag) critical += 1;
        if (i.status !== "RESOLVED") open += 1;
        if (i.needs_human_review && i.status !== "RESOLVED") review += 1;
        if (i.severity_flag && i.status === "NEW") unacknowledgedCritical += 1;
        confidenceSum += i.confidence || 0;
        if (i.assigned_department in byDepartment) byDepartment[i.assigned_department] += 1;

        const ageMinutes = (now - Date.parse(i.received_at)) / 60000;
        if (ageMinutes < RATE_WINDOW_MINUTES) recent += 1;
        const bucket = Math.floor(ageMinutes);
        if (bucket >= 0 && bucket < VOLUME_WINDOW_MINUTES) {
            const slot = volume[VOLUME_WINDOW_MINUTES - 1 - bucket];
            slot.total += 1;
            if (i.severity_flag) slot.critical += 1;
        }
    }

    return {
        total: all.length,
        critical,
        review,
        open,
        unacknowledgedCritical,
        avgConfidence: all.length ? confidenceSum / all.length : null,
        perMinute: recent / RATE_WINDOW_MINUTES,
        byDepartment,
        volume,
    };
}
