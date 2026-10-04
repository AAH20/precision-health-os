"""Integration tests for event bus → alert → dashboard state chain."""

from datetime import datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.dashboard import build_dashboard_state
from precision_health_os.models import Patient, VitalSigns


def _make_patient(patient_id: str = "p001") -> Patient:
    return Patient(
        id=patient_id,
        mrn=f"MRN-{patient_id}",
        name="Test Patient",
        date_of_birth=datetime(1990, 1, 1),
        sex="male",
    )


def _make_critical_vitals(patient_id: str = "p001") -> VitalSigns:
    return VitalSigns(
        patient_id=patient_id,
        heart_rate_bpm=160,
        spo2_percent=80,
    )


class TestEventBusAlertDashboard:
    """Integration tests for the event bus → alert → dashboard chain."""

    def setup_method(self) -> None:
        self.api = PrecisionHealthAPI()

    def test_register_patient_publishes_event(self) -> None:
        patient = _make_patient()
        self.api.register_patient(patient)
        events = self.api.event_bus.get_events("patient.registered")
        assert len(events) == 1
        assert events[0]["payload"]["patient_id"] == patient.id

    def test_ingest_critical_vitals_publishes_alert_event(self) -> None:
        self.api.register_patient(_make_patient())
        self.api.ingest_vitals(_make_critical_vitals())
        events = self.api.event_bus.get_events("alert.generated")
        assert len(events) >= 1
        assert all(e["payload"]["patient_id"] == "p001" for e in events)

    def test_acknowledge_alert_clears_active_and_logs_audit(self) -> None:
        self.api.register_patient(_make_patient())
        self.api.rbac.assign_role("user1", "physician")
        alerts = self.api.ingest_vitals(_make_critical_vitals())
        alert_id = alerts[0].id
        result = self.api.acknowledge_alert(alert_id, "user1")
        assert result is True
        assert len(self.api.get_active_alerts()) == len(alerts) - 1
        assert len(self.api.audit.get_events()) == 1

    def test_dashboard_state_reflects_system_state(self) -> None:
        self.api.register_patient(_make_patient())
        self.api.ingest_vitals(_make_critical_vitals())
        state = build_dashboard_state(self.api)
        assert state["system_status"]["patients_registered"] == 1
        assert state["system_status"]["active_alerts"] >= 1

    def test_dashboard_patient_count(self) -> None:
        self.api.register_patient(_make_patient("p001"))
        self.api.register_patient(_make_patient("p002"))
        state = build_dashboard_state(self.api)
        assert state["system_status"]["patients_registered"] == 2

    def test_dashboard_active_alert_count(self) -> None:
        self.api.register_patient(_make_patient())
        self.api.ingest_vitals(_make_critical_vitals())
        state = build_dashboard_state(self.api)
        assert state["system_status"]["active_alerts"] == len(self.api.get_active_alerts())

    def test_dashboard_audit_event_count(self) -> None:
        self.api.register_patient(_make_patient())
        self.api.rbac.assign_role("user1", "physician")
        self.api.ingest_vitals(_make_critical_vitals())
        alerts = self.api.get_active_alerts()
        self.api.acknowledge_alert(alerts[0].id, "user1")
        state = build_dashboard_state(self.api)
        assert state["system_status"]["audit_events"] == 1

    def test_dashboard_event_bus_event_count(self) -> None:
        self.api.register_patient(_make_patient())
        self.api.ingest_vitals(_make_critical_vitals())
        state = build_dashboard_state(self.api)
        # 1 patient.registered + 2 alert.generated = 3 events
        assert state["system_status"]["event_bus_events"] == 3

    def test_full_flow_register_vitals_alert_acknowledge_dashboard(self) -> None:
        self.api.register_patient(_make_patient())
        self.api.rbac.assign_role("user1", "physician")
        alerts = self.api.ingest_vitals(_make_critical_vitals())
        assert len(alerts) >= 1
        self.api.acknowledge_alert(alerts[0].id, "user1")
        state = build_dashboard_state(self.api)
        assert state["system_status"]["patients_registered"] == 1
        assert state["system_status"]["active_alerts"] == len(alerts) - 1
        assert state["system_status"]["audit_events"] == 1
        assert state["system_status"]["event_bus_events"] == 1 + len(alerts)
