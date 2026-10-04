"""API module: REST API endpoints for Precision Health OS."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from precision_health_os.clinical import CDSSEngine
from precision_health_os.integration import EventBus
from precision_health_os.iot import AnomalyDetector, WearableDataPipeline
from precision_health_os.security import AuditTrail, RBACService

if TYPE_CHECKING:
    from precision_health_os.models import AlertSeverity, ClinicalAlert, Patient, VitalSigns

logger = logging.getLogger(__name__)


class PrecisionHealthAPI:
    """Main API facade for Precision Health OS."""

    def __init__(self) -> None:
        """Initialize the API facade."""
        from precision_health_os.clinical import AlertManager

        self.cds = CDSSEngine()
        self.pipeline = WearableDataPipeline()
        self.detector = AnomalyDetector()
        self.event_bus = EventBus()
        self.audit = AuditTrail()
        self.rbac = RBACService()
        self._patients: dict[str, Patient] = {}
        self._alert_manager = AlertManager()

    def register_patient(self, patient: Patient) -> Patient:
        """Register a new patient."""
        self._patients[patient.id] = patient
        self.event_bus.publish("patient.registered", {"patient_id": patient.id})
        return patient

    def get_patient(self, patient_id: str) -> Patient | None:
        """Get patient by ID."""
        return self._patients.get(patient_id)

    def ingest_vitals(self, vitals: VitalSigns) -> list[ClinicalAlert]:
        """Ingest vital signs and generate alerts."""
        alerts = self.cds.evaluate_vitals(vitals.patient_id, vitals)
        for alert in alerts:
            self._alert_manager.add_alert(alert)
            self.event_bus.publish("alert.generated", alert.model_dump())
        return alerts

    def get_active_alerts(
        self, patient_id: str | None = None, severity: AlertSeverity | None = None
    ) -> list[ClinicalAlert]:
        """Get active alerts."""
        return self._alert_manager.get_active_alerts(patient_id, severity)

    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""
        self.audit.log(user_id, "acknowledge_alert", "alert", alert_id)
        return True

    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check user permission."""
        return self.rbac.has_permission(user_id, permission)

    def get_system_status(self) -> dict[str, Any]:
        """Get system status overview."""
        return {
            "patients_registered": len(self._patients),
            "active_alerts": len(self.get_active_alerts()),
            "audit_events": len(self.audit._events),
            "event_bus_events": len(self.event_bus._event_log),
        }
