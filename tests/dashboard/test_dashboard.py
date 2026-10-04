"""Tests for the dashboard data adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.dashboard import alerts_by_severity, build_dashboard_state
from precision_health_os.models import AlertSeverity, ClinicalAlert, Patient, VitalSigns


def _make_patient(patient_id: str = "P001") -> Patient:
    return Patient(
        id=patient_id,
        mrn=f"MRN-{patient_id}",
        name="Test Patient",
        date_of_birth=datetime(1990, 1, 1, tzinfo=UTC),
        sex="male",
    )


def _make_alert(
    alert_id: str,
    severity: AlertSeverity,
    patient_id: str = "P001",
) -> ClinicalAlert:
    return ClinicalAlert(
        id=alert_id,
        patient_id=patient_id,
        severity=severity,
        title=f"Alert {alert_id}",
        description="Test alert",
        source="test_rule",
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# build_dashboard_state
# ---------------------------------------------------------------------------


class TestBuildDashboardState:
    def test_returns_dict_with_required_keys(self):
        api = PrecisionHealthAPI()
        state = build_dashboard_state(api)
        assert isinstance(state, dict)
        assert "system_status" in state
        assert "modules" in state
        assert "benchmarks" in state
        assert "generated_at" in state

    def test_system_status_contains_expected_fields(self):
        api = PrecisionHealthAPI()
        state = build_dashboard_state(api)
        status = state["system_status"]
        assert "patients_registered" in status
        assert "active_alerts" in status
        assert "audit_events" in status
        assert "event_bus_events" in status

    def test_system_status_values_match_api(self):
        api = PrecisionHealthAPI()
        api.register_patient(_make_patient("P001"))
        api.register_patient(_make_patient("P002"))
        state = build_dashboard_state(api)
        status = state["system_status"]
        assert status["patients_registered"] == 2
        assert status["active_alerts"] == 0
        assert status["audit_events"] == 0
        assert status["event_bus_events"] == 2

    def test_system_status_counts_active_alerts(self):
        api = PrecisionHealthAPI()
        patient = _make_patient("P001")
        api.register_patient(patient)
        vitals = VitalSigns(
            patient_id="P001",
            heart_rate_bpm=200,
            blood_pressure_systolic=180,
            blood_pressure_diastolic=120,
            spo2_percent=85,
            temperature_celsius=39.5,
            respiratory_rate=30,
            glucose_mg_dl=250,
        )
        api.ingest_vitals(vitals)
        state = build_dashboard_state(api)
        status = state["system_status"]
        assert status["active_alerts"] > 0

    def test_system_status_counts_audit_events(self):
        """Acknowledging a real alert records exactly one audit event.

        The original version acknowledged a fabricated id; the API now reports
        False for unknown alerts and does not audit them, so the test must use
        an alert that actually exists.
        """
        from precision_health_os.models import AlertSeverity, ClinicalAlert

        api = PrecisionHealthAPI()
        api.register_patient(_make_patient("P001"))
        alert = api._alert_manager.add_alert(
            ClinicalAlert(
                id="A001",
                patient_id="P001",
                severity=AlertSeverity.HIGH,
                title="test alert",
                description="test",
                source="test",
                confidence=0.9,
            )
        )
        assert api.acknowledge_alert(alert.id, "user1") is True
        state = build_dashboard_state(api)
        status = state["system_status"]
        assert status["audit_events"] == 1

    def test_modules_is_list_of_10_dicts(self):
        api = PrecisionHealthAPI()
        state = build_dashboard_state(api)
        modules = state["modules"]
        assert isinstance(modules, list)
        assert len(modules) == 10
        for mod in modules:
            assert isinstance(mod, dict)
            assert "name" in mod
            assert "purpose" in mod
            assert "status" in mod

    def test_modules_contains_expected_names(self):
        api = PrecisionHealthAPI()
        state = build_dashboard_state(api)
        names = {m["name"] for m in state["modules"]}
        expected = {
            "optimization",
            "clinical",
            "ml",
            "iot",
            "genomics",
            "drug_discovery",
            "security",
            "integration",
            "api",
            "models",
        }
        assert names == expected

    def test_generated_at_is_iso_string(self):
        api = PrecisionHealthAPI()
        state = build_dashboard_state(api)
        ts = state["generated_at"]
        assert isinstance(ts, str)
        # Should be parseable as ISO format
        datetime.fromisoformat(ts)

    def test_state_is_json_serializable(self):
        api = PrecisionHealthAPI()
        api.register_patient(_make_patient("P001"))
        vitals = VitalSigns(
            patient_id="P001",
            heart_rate_bpm=200,
            blood_pressure_systolic=180,
            blood_pressure_diastolic=120,
            spo2_percent=85,
            temperature_celsius=39.5,
            respiratory_rate=30,
            glucose_mg_dl=250,
        )
        api.ingest_vitals(vitals)
        state = build_dashboard_state(api)
        # Must not raise
        json.dumps(state)

    def test_empty_api_produces_valid_state(self):
        api = PrecisionHealthAPI()
        state = build_dashboard_state(api)
        assert state["system_status"]["patients_registered"] == 0
        assert state["system_status"]["active_alerts"] == 0
        assert state["system_status"]["audit_events"] == 0
        assert state["system_status"]["event_bus_events"] == 0
        assert len(state["modules"]) == 10
        json.dumps(state)

    def test_benchmarks_key_exists(self):
        api = PrecisionHealthAPI()
        state = build_dashboard_state(api)
        assert "benchmarks" in state


# ---------------------------------------------------------------------------
# alerts_by_severity
# ---------------------------------------------------------------------------


class TestAlertsBySeverity:
    def test_groups_alerts_by_severity(self):
        alerts = [
            _make_alert("A001", AlertSeverity.CRITICAL),
            _make_alert("A002", AlertSeverity.HIGH),
            _make_alert("A003", AlertSeverity.CRITICAL),
            _make_alert("A004", AlertSeverity.LOW),
        ]
        result = alerts_by_severity(alerts)
        assert result == {"critical": 2, "high": 1, "low": 1}

    def test_empty_list_returns_empty_dict(self):
        result = alerts_by_severity([])
        assert result == {}

    def test_single_alert(self):
        alerts = [_make_alert("A001", AlertSeverity.MEDIUM)]
        result = alerts_by_severity(alerts)
        assert result == {"medium": 1}

    def test_all_severities(self):
        alerts = [
            _make_alert("A001", AlertSeverity.CRITICAL),
            _make_alert("A002", AlertSeverity.HIGH),
            _make_alert("A003", AlertSeverity.MEDIUM),
            _make_alert("A004", AlertSeverity.LOW),
            _make_alert("A005", AlertSeverity.INFO),
        ]
        result = alerts_by_severity(alerts)
        assert result == {
            "critical": 1,
            "high": 1,
            "medium": 1,
            "low": 1,
            "info": 1,
        }

    def test_returns_plain_dict(self):
        alerts = [_make_alert("A001", AlertSeverity.HIGH)]
        result = alerts_by_severity(alerts)
        assert type(result) is dict
