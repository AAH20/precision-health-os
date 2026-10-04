"""Tests for PrecisionHealthAPI.acknowledge_alert edge branches."""

from datetime import datetime

import pytest

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.clinical import AlertManager
from precision_health_os.models import Patient, VitalSigns


@pytest.fixture
def api():
    api = PrecisionHealthAPI()
    api.rbac.assign_role("user1", "physician")
    api.rbac.assign_role("dr_smith", "physician")
    return api


@pytest.fixture
def patient(api):
    p = Patient(
        id="p001",
        mrn="MRN001",
        name="Test Patient",
        date_of_birth=datetime(1990, 1, 1),
        sex="male",
    )
    api.register_patient(p)
    return p


@pytest.fixture
def critical_alert(api, patient):
    """Ingest critical vitals and return the first alert."""
    vitals = VitalSigns(patient_id="p001", heart_rate_bpm=160)
    alerts = api.ingest_vitals(vitals)
    assert len(alerts) >= 1
    return alerts[0]


def test_acknowledge_alert_end_to_end(api, patient, critical_alert):
    """Full flow: alert appears → acknowledge → leaves active list."""
    alert = critical_alert

    active = api.get_active_alerts(patient_id="p001")
    assert alert.id in [a.id for a in active]

    assert api.acknowledge_alert(alert.id, "user1") is True

    active = api.get_active_alerts(patient_id="p001")
    assert alert.id not in [a.id for a in active]


def test_acknowledge_unknown_alert_returns_false(api):
    """Acknowledging a non-existent alert ID returns False."""
    assert api.acknowledge_alert("nonexistent-id", "user1") is False


def test_acknowledge_twice_is_idempotent(api, patient, critical_alert):
    """Acknowledging the same alert twice returns True both times."""
    alert = critical_alert

    assert api.acknowledge_alert(alert.id, "user1") is True
    assert api.acknowledge_alert(alert.id, "user1") is True


def test_audit_trail_records_acknowledge(api, patient, critical_alert):
    """Audit trail records the acknowledge action."""
    alert = critical_alert

    api.acknowledge_alert(alert.id, "dr_smith")

    assert len(api.audit._events) == 1
    event = api.audit._events[0]
    assert event.user_id == "dr_smith"
    assert event.action == "acknowledge_alert"
    assert event.resource_id == alert.id


def test_acknowledge_not_acknowledged_guard(api, patient, critical_alert, monkeypatch):
    """Exercise the `not alert.acknowledged` guard (line 70-73)."""
    alert = critical_alert

    fake_alert = alert.model_copy()
    fake_alert.acknowledged = False
    monkeypatch.setattr(AlertManager, "acknowledge", lambda self, aid, uid: fake_alert)

    assert api.acknowledge_alert(alert.id, "user1") is False
