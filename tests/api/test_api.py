"""Tests for API module."""

from datetime import datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.models import Patient, VitalSigns


class TestPrecisionHealthAPI:
    """Tests for PrecisionHealthAPI."""

    def setup_method(self) -> None:
        self.api = PrecisionHealthAPI()

    def test_register_patient(self) -> None:
        patient = Patient(
            id="p001",
            mrn="MRN001",
            name="Test Patient",
            date_of_birth=datetime(1990, 1, 1),
            sex="male",
        )
        result = self.api.register_patient(patient)
        assert result.id == "p001"
        assert self.api.get_patient("p001") is not None

    def test_get_nonexistent_patient(self) -> None:
        assert self.api.get_patient("nonexistent") is None

    def test_ingest_vitals_normal(self) -> None:
        vitals = VitalSigns(
            patient_id="p001",
            heart_rate_bpm=72,
            spo2_percent=98,
            blood_pressure_systolic=120,
            blood_pressure_diastolic=80,
        )
        alerts = self.api.ingest_vitals(vitals)
        assert len(alerts) == 0

    def test_ingest_vitals_critical(self) -> None:
        vitals = VitalSigns(patient_id="p001", heart_rate_bpm=160)
        alerts = self.api.ingest_vitals(vitals)
        assert len(alerts) >= 1

    def test_get_active_alerts(self) -> None:
        vitals = VitalSigns(patient_id="p001", heart_rate_bpm=160)
        self.api.ingest_vitals(vitals)
        alerts = self.api.get_active_alerts(patient_id="p001")
        assert len(alerts) >= 1

    def test_check_permission(self) -> None:
        self.api.rbac.assign_role("user1", "physician")
        assert self.api.check_permission("user1", "read")
        assert not self.api.check_permission("user1", "admin")

    def test_get_system_status(self) -> None:
        status = self.api.get_system_status()
        assert "patients_registered" in status
        assert "active_alerts" in status
        assert "audit_events" in status
