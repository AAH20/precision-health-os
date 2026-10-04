"""Tests for core data models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from precision_health_os.models import (
    AlertSeverity,
    AuditEvent,
    ClinicalAlert,
    DrugResponsePrediction,
    FHIRResource,
    GenomicVariant,
    Patient,
    TreatmentPlan,
    VitalSigns,
)


class TestPatient:
    """Tests for Patient model."""

    def test_create_patient(self) -> None:
        patient = Patient(
            id="p001",
            mrn="MRN12345",
            name="John Doe",
            date_of_birth=datetime(1990, 1, 1),
            sex="male",
        )
        assert patient.id == "p001"
        assert patient.mrn == "MRN12345"
        assert patient.name == "John Doe"
        assert patient.sex == "male"

    def test_patient_sex_validation(self) -> None:
        with pytest.raises(ValidationError):
            Patient(
                id="p002",
                mrn="MRN12346",
                name="Jane Doe",
                date_of_birth=datetime(1990, 1, 1),
                sex="invalid",
            )

    def test_patient_optional_fields(self) -> None:
        patient = Patient(
            id="p003",
            mrn="MRN12347",
            name="Bob Smith",
            date_of_birth=datetime(1985, 6, 15),
            sex="female",
            weight_kg=70.5,
            height_cm=175.0,
            allergies=["penicillin"],
            medications=["metformin"],
            conditions=["diabetes"],
        )
        assert patient.weight_kg == 70.5
        assert patient.allergies == ["penicillin"]


class TestVitalSigns:
    """Tests for VitalSigns model."""

    def test_create_vitals(self) -> None:
        vitals = VitalSigns(
            patient_id="p001",
            heart_rate_bpm=72,
            spo2_percent=98,
            blood_pressure_systolic=120,
            blood_pressure_diastolic=80,
            temperature_celsius=37.0,
        )
        assert vitals.heart_rate_bpm == 72
        assert vitals.spo2_percent == 98

    def test_vitals_range_validation(self) -> None:
        with pytest.raises(ValidationError):
            VitalSigns(patient_id="p001", heart_rate_bpm=400)

    def test_vitals_optional_fields(self) -> None:
        vitals = VitalSigns(patient_id="p001")
        assert vitals.heart_rate_bpm is None


class TestClinicalAlert:
    """Tests for ClinicalAlert model."""

    def test_create_alert(self) -> None:
        alert = ClinicalAlert(
            id="a001",
            patient_id="p001",
            severity=AlertSeverity.HIGH,
            title="High Heart Rate",
            description="HR > 120 bpm",
            source="vital_thresholds",
            confidence=0.9,
        )
        assert alert.severity == AlertSeverity.HIGH
        assert alert.confidence == 0.9

    def test_alert_confidence_range(self) -> None:
        with pytest.raises(ValidationError):
            ClinicalAlert(
                id="a002",
                patient_id="p001",
                severity=AlertSeverity.LOW,
                title="Test",
                description="Test",
                source="test",
                confidence=1.5,
            )


class TestGenomicVariant:
    """Tests for GenomicVariant model."""

    def test_create_variant(self) -> None:
        variant = GenomicVariant(
            chrom="1",
            pos=12345,
            ref="A",
            alt="G",
            gene="BRCA1",
            clinical_significance="pathogenic",
        )
        assert variant.chrom == "1"
        assert variant.gene == "BRCA1"

    def test_variant_zygosity_validation(self) -> None:
        with pytest.raises(ValidationError):
            GenomicVariant(
                chrom="1",
                pos=12345,
                ref="A",
                alt="G",
                zygosity="invalid",
            )


class TestDrugResponsePrediction:
    """Tests for DrugResponsePrediction model."""

    def test_create_prediction(self) -> None:
        pred = DrugResponsePrediction(
            patient_id="p001",
            drug_name="trastuzumab",
            predicted_response="responder",
            confidence=0.85,
            biomarkers={"HER2": "positive"},
        )
        assert pred.predicted_response == "responder"
        assert pred.confidence == 0.85

    def test_prediction_response_validation(self) -> None:
        with pytest.raises(ValidationError):
            DrugResponsePrediction(
                patient_id="p001",
                drug_name="test",
                predicted_response="invalid",
                confidence=0.5,
            )


class TestTreatmentPlan:
    """Tests for TreatmentPlan model."""

    def test_create_plan(self) -> None:
        plan = TreatmentPlan(
            patient_id="p001",
            diagnosis="breast_cancer",
            medications=[{"name": "trastuzumab", "dose": "6mg/kg"}],
            duration_days=180,
            confidence=0.8,
        )
        assert plan.diagnosis == "breast_cancer"
        assert plan.duration_days == 180


class TestFHIRResource:
    """Tests for FHIRResource model."""

    def test_create_fhir_resource(self) -> None:
        resource = FHIRResource(
            resource_type="Patient",
            id="p001",
            data={"name": [{"text": "John Doe"}]},
        )
        assert resource.resource_type == "Patient"
        assert resource.data["name"][0]["text"] == "John Doe"


class TestAuditEvent:
    """Tests for AuditEvent model."""

    def test_create_audit_event(self) -> None:
        event = AuditEvent(
            id="ae001",
            user_id="user001",
            action="read",
            resource_type="Patient",
            resource_id="p001",
            outcome="success",
        )
        assert event.action == "read"
        assert event.outcome == "success"

    def test_audit_outcome_validation(self) -> None:
        with pytest.raises(ValidationError):
            AuditEvent(
                id="ae002",
                user_id="user001",
                action="read",
                resource_type="Patient",
                resource_id="p001",
                outcome="invalid",
            )
