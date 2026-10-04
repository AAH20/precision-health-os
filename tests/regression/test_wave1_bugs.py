"""Regression tests for two bugs found by Wave 1 coverage agents.

Bug 1: acknowledge_alert logs an audit event on every call, not just the
        first transition. The docstring promises "only on the transition".
Bug 2: EvolutionTracker.history() returns a shallow copy — mutating a dict
        entry in the returned list corrupts the tracker's internal state.
"""

from __future__ import annotations

from datetime import UTC, datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.evaluation import EvolutionTracker
from precision_health_os.models import Patient, VitalSigns


class TestAcknowledgeAlertAuditSemantics:
    """Bug 1: audit event must be logged only on the first transition."""

    def _setup_critical_alert(self) -> tuple[PrecisionHealthAPI, str]:
        api = PrecisionHealthAPI()
        api.rbac.assign_role("dr.smith", "physician")
        patient = Patient(
            id="P-1",
            mrn="MRN-1",
            name="Test",
            date_of_birth=datetime(1990, 1, 1, tzinfo=UTC),
            sex="male",
            weight_kg=70.0,
            height_cm=175.0,
        )
        api.register_patient(patient)
        api.ingest_vitals(
            VitalSigns(
                patient_id="P-1",
                heart_rate_bpm=160,
                spo2_percent=82,
                blood_pressure_systolic=190,
                blood_pressure_diastolic=110,
                temperature_celsius=39.2,
                respiratory_rate=18,
                glucose_mg_dl=95,
            )
        )
        alerts = api.get_active_alerts(patient_id="P-1")
        assert len(alerts) > 0, "expected at least one critical alert"
        return api, alerts[0].id

    def test_first_acknowledge_logs_audit(self) -> None:
        api, alert_id = self._setup_critical_alert()
        result = api.acknowledge_alert(alert_id, "dr.smith")
        assert result is True
        # Exactly one audit event for the acknowledge action
        ack_events = [e for e in api.audit._events if e.action == "acknowledge_alert"]
        assert len(ack_events) == 1, f"expected 1 audit event, got {len(ack_events)}"

    def test_second_acknowledge_does_not_log_audit(self) -> None:
        """Re-acknowledging must not fabricate a second clinical action."""
        api, alert_id = self._setup_critical_alert()
        api.acknowledge_alert(alert_id, "dr.smith")
        events_after_first = len(api.audit._events)

        # Second acknowledge — should be idempotent, no new audit event
        result = api.acknowledge_alert(alert_id, "dr.smith")
        assert result is True  # idempotent success
        events_after_second = len(api.audit._events)
        assert events_after_second == events_after_first, (
            f"re-acknowledge logged {events_after_second - events_after_first} extra audit event(s)"
        )

    def test_unknown_alert_does_not_log_audit(self) -> None:
        api, _ = self._setup_critical_alert()
        events_before = len(api.audit._events)
        result = api.acknowledge_alert("nonexistent-id", "dr.smith")
        assert result is False
        assert len(api.audit._events) == events_before


class TestEvolutionTrackerHistoryImmutability:
    """Bug 2: history() must return a deep copy."""

    def test_mutating_returned_history_does_not_corrupt_tracker(self) -> None:
        tracker = EvolutionTracker(metric="score")
        tracker.record(1, 0.5)
        tracker.record(2, 0.8)

        # Mutate the returned list's dict entry
        h = tracker.history()
        h[0]["score"] = 99.0

        # Internal state must be unaffected
        assert tracker.history()[0]["score"] == 0.5, (
            "mutating returned history corrupted internal state"
        )

    def test_mutating_returned_list_does_not_corrupt_tracker(self) -> None:
        tracker = EvolutionTracker(metric="score")
        tracker.record(1, 0.5)

        h = tracker.history()
        h.clear()

        assert len(tracker.history()) == 1, "clearing returned history corrupted internal state"
