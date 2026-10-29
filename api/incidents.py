# Incident Store
# In-memory record of triaged calls and the dispatcher actions taken on them.
# Shared by every connected dashboard through the API. Data is lost when the server restarts;
# a real deployment would back this with Postgres or Redis.

import uuid
from collections import OrderedDict
from datetime import datetime, timezone

from core.config import ROUTING_MAP, HIGH_SEVERITY_CLASSES

STATUSES = ("NEW", "ACKNOWLEDGED", "DISPATCHED", "RESOLVED")
DEPARTMENT_TO_INDEX = {name: idx for idx, name in ROUTING_MAP.items()}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def is_high_severity(department):
    return DEPARTMENT_TO_INDEX[department] in HIGH_SEVERITY_CLASSES


class IncidentStore:
    def __init__(self, max_size=500):
        self.max_size = max_size
        self._incidents = OrderedDict()

    def create(self, triage):
        now = utc_now()
        incident = {
            "id": f"INC-{uuid.uuid4().hex[:8].upper()}",
            **triage,
            "status": "NEW",
            "model_department": triage["assigned_department"],
            "overridden": False,
            "received_at": now,
            "updated_at": now,
            "notes": [],
            "history": [{
                "at": now,
                "action": "CREATED",
                "detail": f"Model routed to {triage['assigned_department']} ({triage['confidence']:.0%} confidence)",
            }],
        }
        self._incidents[incident["id"]] = incident
        while len(self._incidents) > self.max_size:
            self._incidents.popitem(last=False)
        return incident

    def get(self, incident_id):
        return self._incidents.get(incident_id)

    def recent(self, limit=200):
        """Newest first."""
        return list(reversed(self._incidents.values()))[:limit]

    def update(self, incident_id, status=None, department=None, note=None):
        incident = self._incidents.get(incident_id)
        if incident is None:
            raise KeyError(incident_id)
        if department is not None and department not in DEPARTMENT_TO_INDEX:
            raise ValueError(f"Unknown department: {department}")

        now = utc_now()
        if status is not None and status != incident["status"]:
            incident["history"].append({"at": now, "action": status, "detail": f"{incident['status']} → {status}"})
            incident["status"] = status

        if department is not None and department != incident["assigned_department"]:
            incident["history"].append({
                "at": now, "action": "REROUTED",
                "detail": f"Dispatcher rerouted {incident['assigned_department']} → {department}",
            })
            incident["assigned_department"] = department
            incident["severity_flag"] = is_high_severity(department)
            incident["overridden"] = department != incident["model_department"]

        if note:
            incident["notes"].append({"at": now, "text": note})
            incident["history"].append({"at": now, "action": "NOTE", "detail": note})

        incident["updated_at"] = now
        return incident

    def clear(self):
        self._incidents.clear()
