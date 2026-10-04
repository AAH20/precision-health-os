"""Dashboard data adapter: converts live platform state into JSON-serializable dicts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from precision_health_os.api import PrecisionHealthAPI
    from precision_health_os.models import ClinicalAlert

_MODULES: list[dict[str, str]] = [
    {"name": "optimization", "purpose": "Treatment plan optimization", "status": "active"},
    {"name": "clinical", "purpose": "Clinical decision support", "status": "active"},
    {"name": "ml", "purpose": "Machine learning models", "status": "active"},
    {"name": "iot", "purpose": "IoT/wearable data ingestion", "status": "active"},
    {"name": "genomics", "purpose": "Genomic analysis", "status": "active"},
    {"name": "drug_discovery", "purpose": "Drug response prediction", "status": "active"},
    {"name": "security", "purpose": "Security and access control", "status": "active"},
    {"name": "integration", "purpose": "FHIR/HL7 integration", "status": "active"},
    {"name": "api", "purpose": "REST API endpoints", "status": "active"},
    {"name": "models", "purpose": "Core data models", "status": "active"},
]


def build_dashboard_state(api: PrecisionHealthAPI) -> dict[str, Any]:
    """Build a JSON-serializable dashboard state from live API objects."""
    status = api.get_system_status()
    return {
        "system_status": {
            "patients_registered": status["patients_registered"],
            "active_alerts": status["active_alerts"],
            "audit_events": status["audit_events"],
            "event_bus_events": status["event_bus_events"],
        },
        "modules": [dict(m) for m in _MODULES],
        "benchmarks": {},
        "generated_at": datetime.now(UTC).isoformat(),
    }


def alerts_by_severity(alerts: list[ClinicalAlert]) -> dict[str, int]:
    """Group clinical alerts by severity level."""
    result: dict[str, int] = {}
    for alert in alerts:
        key = alert.severity.value if hasattr(alert.severity, "value") else str(alert.severity)
        result[key] = result.get(key, 0) + 1
    return result
