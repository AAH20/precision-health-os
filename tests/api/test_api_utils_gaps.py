"""Tests for API and utils gap coverage."""

import subprocess
import sys
from datetime import datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.models import AlertSeverity, Patient, VitalSigns
from precision_health_os.utils import (
    chunked,
    deep_merge,
    euclidean_distance,
    generate_token,
    hash_sensitive,
    normalize,
    z_score,
)


def _make_alert(alert_id: str):
    """Build a minimal active ClinicalAlert."""
    from precision_health_os.models import AlertSeverity, ClinicalAlert

    return ClinicalAlert(
        id=alert_id,
        patient_id="p1",
        severity=AlertSeverity.HIGH,
        title=f"alert {alert_id}",
        description="test",
        source="test",
        confidence=0.9,
    )


class TestAcknowledgeAlert:
    """Tests for PrecisionHealthAPI.acknowledge_alert."""

    def setup_method(self) -> None:
        self.api = PrecisionHealthAPI()
        self.api.rbac.assign_role("user-1", "physician")
        self.api.rbac.assign_role("user-456", "physician")

    def test_acknowledge_alert_returns_bool(self) -> None:
        """Acknowledge a real active alert; unknown ids must report False.

        The original version passed a fabricated id and asserted True, which
        matched an implementation that returned True unconditionally without
        ever clearing the alert.
        """
        assert self.api.acknowledge_alert("alert-1", "user-1") is False
        alert = self.api._alert_manager.add_alert(_make_alert("alert-1"))
        assert self.api.acknowledge_alert(alert.id, "user-1") is True

    def test_acknowledge_alert_logs_audit_event(self) -> None:
        alert = self.api._alert_manager.add_alert(_make_alert("alert-123"))
        self.api.acknowledge_alert(alert.id, "user-456")
        events = self.api.audit.get_events(user_id="user-456")
        assert len(events) == 1
        assert events[0].action == "acknowledge_alert"
        assert events[0].resource_id == "alert-123"


class TestGetActiveAlertsSeverityFilter:
    """Tests for get_active_alerts with severity filter."""

    def setup_method(self) -> None:
        self.api = PrecisionHealthAPI()

    def test_get_active_alerts_filter_by_severity(self) -> None:
        # Ingest critical vitals to generate CRITICAL alert
        vitals_critical = VitalSigns(patient_id="p1", heart_rate_bpm=160)
        self.api.ingest_vitals(vitals_critical)

        # Ingest high vitals to generate HIGH alert
        vitals_high = VitalSigns(patient_id="p2", heart_rate_bpm=130)
        self.api.ingest_vitals(vitals_high)

        critical_alerts = self.api.get_active_alerts(severity=AlertSeverity.CRITICAL)
        high_alerts = self.api.get_active_alerts(severity=AlertSeverity.HIGH)

        assert len(critical_alerts) >= 1
        assert len(high_alerts) >= 1
        assert all(a.severity == AlertSeverity.CRITICAL for a in critical_alerts)
        assert all(a.severity == AlertSeverity.HIGH for a in high_alerts)


class TestGenerateToken:
    """Tests for utils.generate_token."""

    def test_generate_token_default_length(self) -> None:
        token = generate_token()
        assert isinstance(token, str)
        assert len(token) > 0

    def test_generate_token_custom_length(self) -> None:
        token = generate_token(64)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_generate_token_uniqueness(self) -> None:
        tokens = {generate_token() for _ in range(100)}
        assert len(tokens) == 100


class TestHashSensitive:
    """Tests for utils.hash_sensitive."""

    def test_hash_sensitive_deterministic(self) -> None:
        hash1 = hash_sensitive("test-value")
        hash2 = hash_sensitive("test-value")
        assert hash1 == hash2

    def test_hash_sensitive_different_values(self) -> None:
        hash1 = hash_sensitive("value1")
        hash2 = hash_sensitive("value2")
        assert hash1 != hash2

    def test_hash_sensitive_across_processes(self) -> None:
        """Verify hash_sensitive produces the same result in a subprocess.

        Uses the installed package (no cwd assumption) so it passes on any
        machine, not just the author's checkout.
        """
        code = (
            "from precision_health_os.utils import hash_sensitive; "
            "print(hash_sensitive('cross-process-test'))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess_hash = result.stdout.strip()
        local_hash = hash_sensitive("cross-process-test")
        assert subprocess_hash == local_hash


class TestDeepMerge:
    """Tests for utils.deep_merge."""

    def test_deep_merge_simple(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self) -> None:
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}


class TestChunked:
    """Tests for utils.chunked."""

    def test_chunked_even_split(self) -> None:
        result = chunked([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_chunked_uneven_split(self) -> None:
        result = chunked([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_chunked_empty(self) -> None:
        result = chunked([], 3)
        assert result == []


class TestZScore:
    """Tests for utils.z_score."""

    def test_z_score_normal(self) -> None:
        result = z_score(10, 5, 2)
        assert result == 2.5

    def test_z_score_zero_std(self) -> None:
        result = z_score(10, 5, 0)
        assert result == 0.0


class TestEuclideanDistance:
    """Tests for utils.euclidean_distance."""

    def test_euclidean_distance_identical(self) -> None:
        result = euclidean_distance([1, 2, 3], [1, 2, 3])
        assert result == 0.0

    def test_euclidean_distance_different(self) -> None:
        result = euclidean_distance([0, 0], [3, 4])
        assert result == 5.0


class TestNormalize:
    """Tests for utils.normalize."""

    def test_normalize_standard(self) -> None:
        result = normalize([1, 2, 3, 4, 5])
        assert result == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_normalize_empty(self) -> None:
        result = normalize([])
        assert result == []

    def test_normalize_same_values(self) -> None:
        result = normalize([5, 5, 5])
        assert result == [0.5, 0.5, 0.5]


class TestEndToEndFlow:
    """End-to-end test: register -> ingest -> alert appears."""

    def test_full_patient_alert_flow(self) -> None:
        api = PrecisionHealthAPI()

        # Register patient
        patient = Patient(
            id="e2e-p1",
            mrn="MRN-E2E",
            name="E2E Patient",
            date_of_birth=datetime(1985, 5, 15),
            sex="female",
        )
        api.register_patient(patient)
        assert api.get_patient("e2e-p1") is not None

        # Ingest critical vitals
        vitals = VitalSigns(patient_id="e2e-p1", heart_rate_bpm=160)
        alerts = api.ingest_vitals(vitals)
        assert len(alerts) >= 1

        # Verify alert appears in active alerts
        active_alerts = api.get_active_alerts(patient_id="e2e-p1")
        assert len(active_alerts) >= 1
