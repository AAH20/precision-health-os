"""Integration tests for API → clinical → alert → audit chain."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.models import AlertSeverity, Patient, VitalSigns


@pytest.fixture
def api():
    api = PrecisionHealthAPI()
    api.rbac.assign_role("user1", "physician")
    return api


def _make_patient(patient_id: str = "P001") -> Patient:
    return Patient(
        id=patient_id,
        mrn=f"MRN-{patient_id}",
        name="Test Patient",
        date_of_birth=datetime(1980, 1, 1, tzinfo=UTC),
        sex="male",
    )


def _critical_vitals(patient_id: str) -> VitalSigns:
    return VitalSigns(
        patient_id=patient_id,
        heart_rate_bpm=160,
        spo2_percent=80,
    )


def test_register_patient_and_ingest_critical_vitals_generates_alerts(api):
    patient = _make_patient()
    api.register_patient(patient)

    vitals = _critical_vitals(patient.id)
    alerts = api.ingest_vitals(vitals)

    assert len(alerts) > 0
    assert all(a.patient_id == patient.id for a in alerts)
    assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)

    active = api.get_active_alerts(patient_id=patient.id)
    assert len(active) == len(alerts)


def test_acknowledge_alert_removes_from_active_list(api):
    patient = _make_patient()
    api.register_patient(patient)

    vitals = _critical_vitals(patient.id)
    alerts = api.ingest_vitals(vitals)
    assert len(alerts) > 0

    alert_id = alerts[0].id
    result = api.acknowledge_alert(alert_id, "user1")
    assert result is True

    active = api.get_active_alerts(patient_id=patient.id)
    assert all(a.id != alert_id for a in active)


def test_audit_chain_intact_after_all_operations(api):
    patient = _make_patient()
    api.register_patient(patient)

    vitals = _critical_vitals(patient.id)
    api.ingest_vitals(vitals)

    alerts = api.get_active_alerts(patient_id=patient.id)
    for alert in alerts:
        api.acknowledge_alert(alert.id, "user1")

    assert api.audit.verify_chain() is True


def test_audit_events_logged_for_each_action(api):
    patient = _make_patient()
    api.register_patient(patient)

    vitals = _critical_vitals(patient.id)
    api.ingest_vitals(vitals)

    alerts = api.get_active_alerts(patient_id=patient.id)
    for alert in alerts:
        api.acknowledge_alert(alert.id, "user1")

    events = api.audit.get_events()
    ack_events = [e for e in events if e.action == "acknowledge_alert"]
    assert len(ack_events) == len(alerts)


def test_full_flow_multiple_patients_simultaneously(api):
    patients = [_make_patient(f"P{i:03d}") for i in range(3)]
    for p in patients:
        api.register_patient(p)

    all_alerts = {}
    for p in patients:
        vitals = _critical_vitals(p.id)
        alerts = api.ingest_vitals(vitals)
        all_alerts[p.id] = alerts

    for p in patients:
        active = api.get_active_alerts(patient_id=p.id)
        assert len(active) == len(all_alerts[p.id])

    for alert in all_alerts[patients[0].id]:
        api.acknowledge_alert(alert.id, "user1")

    assert len(api.get_active_alerts(patient_id=patients[0].id)) == 0

    for p in patients[1:]:
        assert len(api.get_active_alerts(patient_id=p.id)) == len(all_alerts[p.id])

    assert api.audit.verify_chain() is True


def test_acknowledge_unknown_alert_returns_false_and_logs_nothing(api):
    patient = _make_patient()
    api.register_patient(patient)

    vitals = _critical_vitals(patient.id)
    api.ingest_vitals(vitals)

    events_before = len(api.audit.get_events())

    result = api.acknowledge_alert("nonexistent-alert-id", "user1")
    assert result is False

    events_after = len(api.audit.get_events())
    assert events_after == events_before


def test_acknowledge_already_acknowledged_alert_is_idempotent(api):
    patient = _make_patient()
    api.register_patient(patient)

    vitals = _critical_vitals(patient.id)
    alerts = api.ingest_vitals(vitals)
    assert len(alerts) > 0

    alert_id = alerts[0].id
    result1 = api.acknowledge_alert(alert_id, "user1")
    assert result1 is True

    events_after_first = len(api.audit.get_events())

    result2 = api.acknowledge_alert(alert_id, "user1")
    assert result2 is True

    events_after_second = len(api.audit.get_events())
    assert events_after_second == events_after_first
