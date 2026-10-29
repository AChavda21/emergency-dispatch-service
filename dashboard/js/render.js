// DOM rendering. Every piece of incident data is inserted with textContent
// (never innerHTML) so a malicious transcript cannot inject markup or script.

import { departmentMeta, STATUS_LABELS, VOLUME_WINDOW_MINUTES } from "./config.js";

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const MAX_RENDERED_ROWS = 200;

export function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (value === null || value === undefined || value === false) continue;
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = value;
        else if (key === "style") Object.entries(value).forEach(([p, v]) => node.style.setProperty(p, v));
        else if (key === "dataset") Object.assign(node.dataset, value);
        else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) {
        if (child === null || child === undefined || child === false) continue;
        node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
}

// ---------- Formatting ----------

export function formatTime(iso) {
    return new Date(iso).toLocaleTimeString([], { hour12: false });
}

export function formatRelative(iso, now = Date.now()) {
    const seconds = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
    if (seconds < 10) return "just now";
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m ago`;
}

export function formatPercent(value) {
    return value === null || value === undefined ? "—" : `${Math.round(value * 100)}%`;
}

function formatUptime(seconds) {
    const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60);
    return h ? `${h}h ${m}m` : `${m}m`;
}

// ---------- Small components ----------

function deptBadge(department) {
    return el("span", { class: "badge badge-dept", style: { "--dept": departmentMeta(department).color }, text: department });
}

function statusTag(status) {
    return el("span", { class: "badge status-tag", dataset: { status }, text: STATUS_LABELS[status] || status });
}

function confidenceMeter(confidence, threshold) {
    const low = confidence < threshold;
    return el("span", { class: "meter", "data-low": String(low), title: `Model confidence ${formatPercent(confidence)}` },
        el("span", { class: "meter-track" }, el("span", { class: "meter-fill", style: { width: `${confidence * 100}%` } })),
        el("span", { text: formatPercent(confidence) }),
    );
}

function incidentBadges(incident) {
    return [
        deptBadge(incident.assigned_department),
        incident.severity_flag && el("span", { class: "badge badge-critical", text: "High" }),
        incident.needs_human_review && el("span", { class: "badge badge-review", text: "Review" }),
        statusTag(incident.status),
    ];
}

// ---------- Top bar ----------

export function renderHealth(health, socket) {
    const model = $("health-model");
    const sock = $("health-socket");
    const device = $("health-device");

    if (!health) {
        model.dataset.state = "down";
        model.textContent = "API unreachable";
    } else if (health.model_loaded) {
        model.dataset.state = "ok";
        model.textContent = `Model ${health.model_path}`;
    } else {
        model.dataset.state = "down";
        model.textContent = "Model offline";
    }

    device.dataset.state = health ? "ok" : "warn";
    device.textContent = health ? `${health.device.toUpperCase()} · up ${formatUptime(health.uptime_seconds)}` : "—";

    const socketLabels = { open: ["ok", "Live feed"], connecting: ["warn", "Connecting…"], closed: ["down", "Reconnecting…"] };
    const [socketState, socketText] = socketLabels[socket] || socketLabels.closed;
    sock.dataset.state = socketState;
    sock.textContent = socketText;
    $("live-dot").dataset.state = socket === "open" ? "ok" : "down";

    const banner = $("banner");
    const offline = !health || !health.model_loaded;
    banner.classList.toggle("is-visible", offline);
    banner.textContent = !health
        ? "Cannot reach the triage API. Retrying automatically. Calls will not be triaged until the connection is restored."
        : "The triage model is not loaded. New calls will be rejected (503). Check the API server logs.";
}

// ---------- Sidebar ----------

export function renderStats(stats) {
    $("kpi-total").textContent = stats.total;
    $("kpi-critical").textContent = stats.critical;
    $("kpi-review").textContent = stats.review;
    $("kpi-open").textContent = stats.open;
    $("kpi-confidence").textContent = formatPercent(stats.avgConfidence);
    $("kpi-rate").textContent = stats.perMinute.toFixed(1);
}

export function renderDepartments(stats, departments, activeDepartment) {
    const max = Math.max(1, ...Object.values(stats.byDepartment));
    $("dept-bars").replaceChildren(...departments.map((dept) => {
        const count = stats.byDepartment[dept] || 0;
        return el("button", {
            class: "dept-row", type: "button",
            style: { "--dept": departmentMeta(dept).color },
            "aria-pressed": String(activeDepartment === dept),
            title: `Filter feed to ${dept}`,
            dataset: { dept },
        },
            el("span", { class: "dept-name", text: dept }),
            el("span", { class: "dept-count", text: count }),
            el("span", { class: "bar" }, el("span", { class: "bar-fill", style: { width: `${(count / max) * 100}%` } })),
        );
    }));
}

export function renderVolume(stats) {
    const svg = $("volume-chart");
    const width = 300, height = 64, gap = 2;
    const barWidth = width / VOLUME_WINDOW_MINUTES - gap;
    const max = Math.max(1, ...stats.volume.map((v) => v.total));

    const bars = [];
    stats.volume.forEach((slot, i) => {
        const x = i * (barWidth + gap);
        const totalHeight = (slot.total / max) * (height - 4);
        const criticalHeight = (slot.critical / max) * (height - 4);
        const title = `${VOLUME_WINDOW_MINUTES - i - 1} min ago: ${slot.total} calls (${slot.critical} high severity)`;
        for (const [h, cls] of [[totalHeight, "sparkline-bar"], [criticalHeight, "sparkline-bar is-critical"]]) {
            if (!h) continue;
            const rect = document.createElementNS(SVG_NS, "rect");
            rect.setAttribute("x", x.toFixed(1));
            rect.setAttribute("y", (height - h).toFixed(1));
            rect.setAttribute("width", barWidth.toFixed(1));
            rect.setAttribute("height", h.toFixed(1));
            rect.setAttribute("rx", "1");
            rect.setAttribute("class", cls);
            const tip = document.createElementNS(SVG_NS, "title");
            tip.textContent = title;
            rect.append(tip);
            bars.push(rect);
        }
    });
    svg.replaceChildren(...bars);
}

export function renderShortcuts(shortcuts) {
    $("shortcuts").replaceChildren(...shortcuts.flatMap(([key, label]) => [
        el("span", {}, ...key.split(" / ").flatMap((k, i) => [i ? " / " : null, el("span", { class: "kbd", text: k })])),
        el("span", { text: label }),
    ]));
}

// ---------- Feed ----------

export function renderFeed({ incidents, total, selectedId, freshIds, hasFilters, threshold, now }) {
    const list = $("feed-list");
    $("feed-count").textContent = hasFilters ? `${incidents.length} of ${total}` : `${total}`;
    $("btn-clear-filters").hidden = !hasFilters;

    if (!incidents.length) {
        list.replaceChildren(emptyState(
            hasFilters ? "No matching calls" : "Awaiting telemetry…",
            hasFilters ? "Try clearing the filters." : "Calls appear here the moment they are triaged.",
        ));
        $("feed-footer-left").textContent = "";
        return;
    }

    const rows = incidents.slice(0, MAX_RENDERED_ROWS).map((incident) => {
        const classes = ["incident", `status-${incident.status}`];
        if (incident.severity_flag) classes.push("is-critical");
        if (incident.status === "RESOLVED") classes.push("is-resolved");
        if (freshIds.has(incident.id)) classes.push("is-new");

        return el("button", {
            class: classes.join(" "),
            type: "button",
            role: "option",
            "aria-selected": String(incident.id === selectedId),
            dataset: { id: incident.id },
        },
            el("div", { class: "incident-head" },
                ...incidentBadges(incident),
                el("span", { class: "incident-meta" },
                    el("span", { text: incident.caller_id }),
                    el("span", { text: formatTime(incident.received_at) }),
                ),
            ),
            el("div", { class: "incident-text", text: incident.transcript }),
            el("div", { class: "incident-foot" },
                confidenceMeter(incident.confidence, threshold),
                el("span", { text: incident.id }),
                incident.overridden && el("span", { text: `rerouted from ${incident.model_department}` }),
                el("span", { style: { "margin-left": "auto" }, text: formatRelative(incident.received_at, now) }),
            ),
        );
    });

    list.replaceChildren(...rows);
    freshIds.clear();
    $("feed-footer-left").textContent = incidents.length > MAX_RENDERED_ROWS
        ? `Showing newest ${MAX_RENDERED_ROWS} of ${incidents.length}` : "";
}

function emptyState(title, hint) {
    const icon = document.createElementNS(SVG_NS, "svg");
    icon.setAttribute("viewBox", "0 0 24 24");
    icon.setAttribute("fill", "none");
    icon.setAttribute("stroke", "currentColor");
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("stroke-width", "1.2");
    path.setAttribute("stroke-linejoin", "round");
    path.setAttribute("d", "M13 10V3L4 14h7v7l9-11h-7z");
    icon.append(path);
    return el("div", { class: "empty" }, icon,
        el("div", { class: "empty-title", text: title }),
        el("div", { class: "empty-hint", text: hint }));
}

// ---------- Detail panel ----------

export function renderDetail(incident, config, now) {
    const panel = $("detail");
    const body = $("detail-body");
    panel.classList.toggle("is-open", Boolean(incident));

    // Keep a half-typed note if the same incident re-renders because of a live update
    const noteInput = $("note-input");
    const draft = noteInput && noteInput.dataset.incident === (incident && incident.id) ? noteInput.value : "";
    const noteFocused = noteInput && document.activeElement === noteInput;

    if (!incident) {
        body.replaceChildren(emptyState("No call selected", "Select a call from the feed, or press j to jump to the newest."));
        return;
    }

    const probabilities = Object.entries(incident.probabilities || {}).sort((a, b) => b[1] - a[1]);
    const isResolved = incident.status === "RESOLVED";

    const statusButtons = [
        ["ACKNOWLEDGED", "Acknowledge", "a"],
        ["DISPATCHED", "Dispatch", "d"],
        ["RESOLVED", "Resolve", "r"],
    ].map(([status, label, key]) => el("button", {
        class: "btn btn-status", type: "button",
        dataset: { action: "status", status },
        disabled: incident.status === status,
    }, label, el("span", { class: "kbd", text: key })));

    const reroute = el("select", { class: "select", id: "reroute-select", "aria-label": "Reroute to department" },
        ...config.departments.map((d) => el("option", { value: d, selected: d === incident.assigned_department, text: d })));

    const note = el("textarea", {
        class: "textarea", id: "note-input", rows: "2", maxlength: "500",
        placeholder: "Add a note (unit assigned, caller callback, …)",
        dataset: { incident: incident.id },
    });
    note.value = draft;

    body.replaceChildren(
        el("div", { class: "detail-header" },
            el("div", { class: "detail-title-row" },
                el("span", { class: "detail-id", text: incident.id }),
                el("button", { class: "btn btn-ghost detail-close", type: "button", dataset: { action: "close" }, "aria-label": "Close details" }, "✕"),
            ),
            el("div", { class: "detail-badges" },
                ...incidentBadges(incident),
                incident.overridden && el("span", { class: "badge badge-override", text: "Rerouted" }),
            ),
            el("div", { class: "detail-times",
                text: `Caller ${incident.caller_id} · received ${formatTime(incident.received_at)} (${formatRelative(incident.received_at, now)})` }),
        ),

        el("section", {},
            el("div", { class: "section-label", text: "Transcript" }),
            el("div", { class: "transcript", text: incident.transcript }),
        ),

        el("section", {},
            el("div", { class: "section-label", text: "Model assessment" }),
            el("div", { class: "prob-list" }, ...probabilities.map(([dept, p], i) => el("div", {
                class: `prob-row${i === 0 ? " is-top" : ""}`, style: { "--dept": departmentMeta(dept).color },
            },
                el("span", { class: "dept-name", text: dept }),
                el("span", { class: "bar" }, el("span", { class: "bar-fill", style: { width: `${p * 100}%` } })),
                el("span", { class: "prob-value", text: formatPercent(p) }),
            ))),
            incident.needs_human_review && el("p", { class: "field-hint",
                text: `Confidence is below the ${formatPercent(config.confidence_threshold)} threshold. Verify the department with the caller before dispatching.` }),
            incident.overridden && el("p", { class: "field-hint", text: `Model originally routed this call to ${incident.model_department}.` }),
        ),

        el("section", {},
            el("div", { class: "section-label", text: "Actions" }),
            el("div", { class: "actions" }, ...statusButtons),
            isResolved && el("div", { class: "action-row" },
                el("button", { class: "btn btn-status", type: "button", dataset: { action: "status", status: "NEW" } }, "Reopen")),
            el("div", { class: "action-row" }, reroute,
                el("button", { class: "btn", type: "button", dataset: { action: "reroute" } }, "Reroute")),
        ),

        el("section", { class: "note-form" },
            el("label", { class: "section-label", for: "note-input", text: "Notes" }),
            note,
            el("button", { class: "btn", type: "button", dataset: { action: "note" } }, "Add note"),
        ),

        el("section", {},
            el("div", { class: "section-label", text: "Timeline" }),
            el("ol", { class: "timeline" }, ...[...incident.history].reverse().map((entry) => el("li", {},
                el("span", { class: "timeline-action", text: entry.action }),
                el("span", { class: "timeline-time", text: formatTime(entry.at) }),
                el("div", { class: "timeline-detail", text: entry.detail }),
            ))),
        ),
    );

    if (noteFocused) {
        note.focus();
        note.setSelectionRange(note.value.length, note.value.length);
    }
}

// ---------- Toasts ----------

export function toast(title, body = "", tone = "info", timeout = 5000) {
    const node = el("div", { class: "toast", role: tone === "danger" ? "alert" : "status", dataset: { tone } },
        el("div", { class: "toast-title", text: title }),
        body && el("div", { class: "toast-body", text: body }),
    );
    const container = $("toasts");
    container.prepend(node);
    while (container.children.length > 4) container.lastChild.remove();
    setTimeout(() => {
        node.classList.add("is-leaving");
        setTimeout(() => node.remove(), 200);
    }, timeout);
}
