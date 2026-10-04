"""Tests for clinical module."""


from precision_health_os.clinical import (
    AlertManager,
    CDSSEngine,
    ClinicalPathwayOptimizer,
    ClinicalRule,
)
from precision_health_os.models import AlertSeverity, ClinicalAlert, VitalSigns


class TestCDSSEngine:
    """Tests for CDSS Engine."""

    def setup_method(self) -> None:
        self.engine = CDSSEngine()

    def test_normal_vitals_no_alerts(self) -> None:
        vitals = VitalSigns(
            patient_id="p001",
            heart_rate_bpm=72,
            spo2_percent=98,
            blood_pressure_systolic=120,
            blood_pressure_diastolic=80,
            temperature_celsius=37.0,
        )
        alerts = self.engine.evaluate_vitals("p001", vitals)
        assert len(alerts) == 0

    def test_critical_heart_rate_high(self) -> None:
        vitals = VitalSigns(patient_id="p001", heart_rate_bpm=160)
        alerts = self.engine.evaluate_vitals("p001", vitals)
        assert len(alerts) >= 1
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)

    def test_critical_spo2_low(self) -> None:
        vitals = VitalSigns(patient_id="p001", spo2_percent=82)
        alerts = self.engine.evaluate_vitals("p001", vitals)
        assert len(alerts) >= 1
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)

    def test_high_blood_pressure(self) -> None:
        vitals = VitalSigns(
            patient_id="p001",
            blood_pressure_systolic=190,
            blood_pressure_diastolic=115,
        )
        alerts = self.engine.evaluate_vitals("p001", vitals)
        assert len(alerts) >= 1

    def test_add_custom_rule(self) -> None:
        rule = ClinicalRule(
            id="rule001",
            name="Test Rule",
            condition="True",
            severity=AlertSeverity.LOW,
            message="Test alert",
        )
        self.engine.add_rule(rule)
        assert "rule001" in self.engine._rules

    def test_remove_rule(self) -> None:
        rule = ClinicalRule(
            id="rule002",
            name="Test Rule 2",
            condition="True",
            severity=AlertSeverity.LOW,
            message="Test alert 2",
        )
        self.engine.add_rule(rule)
        self.engine.remove_rule("rule002")
        assert "rule002" not in self.engine._rules


class TestAlertManager:
    """Tests for AlertManager."""

    def setup_method(self) -> None:
        self.manager = AlertManager()

    def test_add_alert(self) -> None:
        alert = ClinicalAlert(
            id="a001",
            patient_id="p001",
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="Test",
            source="test",
            confidence=0.9,
        )
        result = self.manager.add_alert(alert)
        assert result.id == "a001"

    def test_deduplication(self) -> None:
        alert1 = ClinicalAlert(
            id="a001",
            patient_id="p001",
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="Test",
            source="test",
            confidence=0.9,
        )
        alert2 = ClinicalAlert(
            id="a002",
            patient_id="p001",
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="Test",
            source="test",
            confidence=0.9,
        )
        self.manager.add_alert(alert1)
        result = self.manager.add_alert(alert2)
        assert result.id == "a001"  # Returns original

    def test_acknowledge(self) -> None:
        alert = ClinicalAlert(
            id="a001",
            patient_id="p001",
            severity=AlertSeverity.HIGH,
            title="Test",
            description="Test",
            source="test",
            confidence=0.9,
        )
        self.manager.add_alert(alert)
        self.manager.acknowledge("a001", "user1")
        active = self.manager.get_active_alerts()
        assert len(active) == 0

    def test_get_active_alerts_filter(self) -> None:
        alert1 = ClinicalAlert(
            id="a001",
            patient_id="p001",
            severity=AlertSeverity.HIGH,
            title="Alert 1",
            description="Test",
            source="test",
            confidence=0.9,
        )
        alert2 = ClinicalAlert(
            id="a002",
            patient_id="p002",
            severity=AlertSeverity.LOW,
            title="Alert 2",
            description="Test",
            source="test",
            confidence=0.5,
        )
        self.manager.add_alert(alert1)
        self.manager.add_alert(alert2)

        p1_alerts = self.manager.get_active_alerts(patient_id="p001")
        assert len(p1_alerts) == 1

        high_alerts = self.manager.get_active_alerts(severity=AlertSeverity.HIGH)
        assert len(high_alerts) == 1


class TestClinicalPathwayOptimizer:
    """Tests for ClinicalPathwayOptimizer."""

    def test_register_pathway(self) -> None:
        optimizer = ClinicalPathwayOptimizer()
        steps = [
            {"name": "step1", "condition": "True"},
            {"name": "step2", "condition": "False"},
        ]
        optimizer.register_pathway("test_pathway", steps)
        assert "test_pathway" in optimizer._pathways

    def test_optimize_pathway(self) -> None:
        optimizer = ClinicalPathwayOptimizer()
        steps = [
            {"name": "step1", "condition": "True"},
            {"name": "step2", "condition": "False"},
            {"name": "step3"},
        ]
        optimizer.register_pathway("test_pathway", steps)
        result = optimizer.optimize("test_pathway", {})
        assert len(result) == 2  # step1 (True) + step3 (no condition)

    def test_unknown_pathway(self) -> None:
        optimizer = ClinicalPathwayOptimizer()
        try:
            optimizer.optimize("unknown", {})
            assert False, "Should raise ValueError"
        except ValueError:
            pass
