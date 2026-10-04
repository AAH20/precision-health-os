"""Regression tests for two bugs found during the coverage audit.

Bug A (iot): ``AnomalyDetector.detect_iqr`` raised ``ZeroDivisionError`` when
the interquartile range was zero — a constant baseline with a single outlier,
e.g. ``[70, 70, 70, 70, 120]``. A crash in the monitoring pipeline means no
alert is raised at all for the very excursion that matters most.

Bug B (api): ``PrecisionHealthAPI.acknowledge_alert`` logged an audit event
and returned ``True`` but never called the alert manager's ``acknowledge``,
so the alert stayed active. A clinician clearing an alert would see it
reappear — the API reported success it did not deliver.
"""

from __future__ import annotations

from datetime import datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.iot import AnomalyDetector
from precision_health_os.models import Patient, VitalSigns


class TestIQRZeroRange:
    """detect_iqr must not divide by zero when the IQR is degenerate."""

    def test_constant_baseline_with_outlier_does_not_raise(self) -> None:
        """A flat baseline plus one spike is the classic excursion case."""
        detector = AnomalyDetector()
        results = detector.detect_iqr([70, 70, 70, 70, 120])
        assert len(results) == 5

    def test_outlier_is_flagged_when_iqr_is_zero(self) -> None:
        """Degenerate range must still catch the outlier, not silently pass it."""
        detector = AnomalyDetector()
        results = detector.detect_iqr([70, 70, 70, 70, 120])
        assert results[-1].is_anomaly is True
        assert results[0].is_anomaly is False

    def test_all_constant_series_is_never_flagged(self) -> None:
        """No spread and no outlier means nothing to report."""
        detector = AnomalyDetector()
        results = detector.detect_iqr([70, 70, 70, 70, 70])
        assert not any(r.is_anomaly for r in results)

    def test_score_is_finite_when_iqr_is_zero(self) -> None:
        """The score must be a finite number, not inf/nan."""
        import math

        detector = AnomalyDetector()
        for r in detector.detect_iqr([70, 70, 70, 70, 120]):
            assert math.isfinite(r.score), f"non-finite score: {r.score}"

    def test_ensemble_survives_zero_iqr(self) -> None:
        """The crash must not propagate through the ensemble path.

        The ensemble verdict itself is not asserted here: on a zero-variance
        baseline the z-score arm divides by a zero standard deviation and the
        EWMA arm is in warm-up, so the ensemble's job is to not crash. The
        detector-level guarantee is covered by TestIQRZeroRange.
        """
        detector = AnomalyDetector()
        results = detector.detect_ensemble([70, 70, 70, 70, 120])
        assert len(results) == 5
        assert results[-1].details["iqr"] is True

    def test_normal_iqr_still_detects_as_before(self) -> None:
        """Non-degenerate ranges must keep working."""
        detector = AnomalyDetector()
        results = detector.detect_iqr([10, 11, 9, 10, 10, 11, 9, 100])
        assert results[-1].is_anomaly


class TestAcknowledgeAlertActuallyAcknowledges:
    """acknowledge_alert must clear the alert, not just log."""

    def _api_with_active_alert(self) -> tuple[PrecisionHealthAPI, str]:
        api = PrecisionHealthAPI()
        api.register_patient(
            Patient(
                id="p1",
                mrn="M1",
                name="Test",
                date_of_birth=datetime(1990, 1, 1),
                sex="male",
            )
        )
        api.ingest_vitals(VitalSigns(patient_id="p1", heart_rate_bpm=160))
        active = api.get_active_alerts(patient_id="p1")
        assert active, "precondition: a critical alert should be active"
        return api, active[0].id

    def test_acknowledged_alert_leaves_active_list(self) -> None:
        api, alert_id = self._api_with_active_alert()
        assert api.acknowledge_alert(alert_id, "dr_smith") is True
        remaining = api.get_active_alerts(patient_id="p1")
        assert all(a.id != alert_id for a in remaining), "alert still active after ack"

    def test_active_list_shrinks_by_one(self) -> None:
        api, alert_id = self._api_with_active_alert()
        before = len(api.get_active_alerts(patient_id="p1"))
        api.acknowledge_alert(alert_id, "dr_smith")
        after = len(api.get_active_alerts(patient_id="p1"))
        assert after == before - 1

    def test_audit_event_still_recorded(self) -> None:
        """Fixing the state change must not lose the audit trail."""
        api, alert_id = self._api_with_active_alert()
        api.acknowledge_alert(alert_id, "dr_smith")
        events = api.audit.get_events(user_id="dr_smith")
        assert len(events) == 1
        assert events[0].action == "acknowledge_alert"
        assert events[0].resource_id == alert_id

    def test_unknown_alert_id_returns_false(self) -> None:
        """Acknowledging a nonexistent alert must not report success."""
        api, _ = self._api_with_active_alert()
        assert api.acknowledge_alert("does-not-exist", "dr_smith") is False

    def test_acknowledge_is_idempotent(self) -> None:
        """Re-acknowledging is idempotent and does not fabricate audit events.

        The audit event is recorded only on the transition, so a second
        acknowledge call returns True but does not log a duplicate event.
        """
        api, alert_id = self._api_with_active_alert()
        assert api.acknowledge_alert(alert_id, "dr_smith") is True
        assert api.acknowledge_alert(alert_id, "dr_smith") is True
        assert api.get_active_alerts(patient_id="p1") == []
        assert len(api.audit.get_events(user_id="dr_smith")) == 1


class TestAPIAuditTrailIntegrity:
    """The audit trail must stay verifiable after acknowledgement."""

    def test_audit_chain_remains_valid(self) -> None:
        api = PrecisionHealthAPI()
        api.register_patient(
            Patient(
                id="p1",
                mrn="M1",
                name="T",
                date_of_birth=datetime(1990, 1, 1),
                sex="male",
            )
        )
        api.ingest_vitals(VitalSigns(patient_id="p1", heart_rate_bpm=160))
        for a in api.get_active_alerts(patient_id="p1"):
            api.acknowledge_alert(a.id, "dr_smith")
        assert api.audit.verify_chain() is True
