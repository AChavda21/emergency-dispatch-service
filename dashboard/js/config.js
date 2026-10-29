// Display metadata for the dashboard. Routing rules themselves come from GET /v1/config,
// these values are only the fallback used before (or if) that request succeeds.

export const DEPARTMENT_META = {
    "MEDICAL":           { color: "var(--dept-medical)",    short: "MED" },
    "FIRE":              { color: "var(--dept-fire)",       short: "FIRE" },
    "VIOLENT CRIME":     { color: "var(--dept-violent)",    short: "PD-V" },
    "NON-VIOLENT CRIME": { color: "var(--dept-nonviolent)", short: "PD-NV" },
    "RESCUE & HAZMAT":   { color: "var(--dept-hazmat)",     short: "HAZ" },
};

export const DEFAULT_CONFIG = {
    departments: Object.keys(DEPARTMENT_META),
    high_severity_departments: ["FIRE", "VIOLENT CRIME", "RESCUE & HAZMAT"],
    statuses: ["NEW", "ACKNOWLEDGED", "DISPATCHED", "RESOLVED"],
    confidence_threshold: 0.6,
};

export const STATUS_LABELS = {
    NEW: "New",
    ACKNOWLEDGED: "Acknowledged",
    DISPATCHED: "Dispatched",
    RESOLVED: "Resolved",
};

export const MAX_CLIENT_INCIDENTS = 500;
export const VOLUME_WINDOW_MINUTES = 30;
export const RATE_WINDOW_MINUTES = 5;
export const HEALTH_POLL_MS = 15000;
export const RELATIVE_TIME_REFRESH_MS = 15000;

export const SAMPLE_CALLS = [
    "Emergency at Whyte Ave, there is a man with a knife threatening people. Send units!",
    "There's thick black smoke coming from the apartment on the 3rd floor, please hurry!",
    "My grandfather just collapsed and he is not breathing. Please send help now!",
    "Some kids just smashed my car windows in the parking lot at Southgate Mall.",
    "Massive car crash on the Yellowhead Trail, someone is trapped in the vehicle!",
];

export const SHORTCUTS = [
    ["/", "Search"],
    ["j / k", "Next / previous call"],
    ["a", "Acknowledge"],
    ["d", "Dispatch"],
    ["r", "Resolve"],
    ["n", "New call"],
    ["m", "Toggle audio"],
    ["Esc", "Close / clear"],
];

export function departmentMeta(name) {
    return DEPARTMENT_META[name] || { color: "var(--text-muted)", short: name };
}
