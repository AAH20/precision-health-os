"""Tests for ML module."""

from datetime import datetime

from precision_health_os.ml import DiseasePredictor, DrugResponseModel, MedicalImageAnalyzer
from precision_health_os.models import Patient


class TestDiseasePredictor:
    """Tests for DiseasePredictor."""

    def setup_method(self) -> None:
        self.predictor = DiseasePredictor()
        self.patient = Patient(
            id="p001",
            mrn="MRN001",
            name="Test Patient",
            date_of_birth=datetime(1970, 1, 1),
            sex="male",
        )

    def test_cardiovascular_risk_low(self) -> None:
        vitals = {"systolic_bp": 120, "total_cholesterol": 180, "hdl": 55, "glucose": 90, "bmi": 24}
        result = self.predictor.predict_cardiovascular_risk(self.patient, vitals)
        assert result.disease == "cardiovascular_disease"
        assert result.risk_score < 0.3

    def test_cardiovascular_risk_high(self) -> None:
        vitals = {"systolic_bp": 160, "total_cholesterol": 260, "hdl": 35, "glucose": 140, "bmi": 32}
        result = self.predictor.predict_cardiovascular_risk(self.patient, vitals)
        assert result.risk_score > 0.3
        assert len(result.contributing_factors) > 0

    def test_diabetes_risk_low(self) -> None:
        labs = {"fasting_glucose": 90, "hba1c": 5.0, "bmi": 22}
        result = self.predictor.predict_diabetes_risk(self.patient, labs)
        assert result.disease == "type_2_diabetes"
        assert result.risk_score < 0.3

    def test_diabetes_risk_high(self) -> None:
        labs = {"fasting_glucose": 140, "hba1c": 7.0, "bmi": 32}
        result = self.predictor.predict_diabetes_risk(self.patient, labs)
        assert result.risk_score > 0.3


class TestDrugResponseModel:
    """Tests for DrugResponseModel."""

    def setup_method(self) -> None:
        self.model = DrugResponseModel()

    def test_responder_prediction(self) -> None:
        biomarkers = {"HER2": "positive"}
        result = self.model.predict_response("p001", "trastuzumab", biomarkers)
        assert result.predicted_response == "responder"
        assert result.confidence > 0.5

    def test_non_responder_prediction(self) -> None:
        biomarkers = {"HER2": "negative"}
        result = self.model.predict_response("p001", "trastuzumab", biomarkers)
        assert result.predicted_response == "non-responder"

    def test_kras_resistance(self) -> None:
        biomarkers = {"KRAS": "mutated"}
        result = self.model.predict_response("p001", "cetuximab", biomarkers)
        assert result.predicted_response == "non-responder"


class TestMedicalImageAnalyzer:
    """Tests for MedicalImageAnalyzer."""

    def setup_method(self) -> None:
        self.analyzer = MedicalImageAnalyzer()

    def test_analyze_chest_xray(self) -> None:
        result = self.analyzer.analyze_chest_xray(b"fake_image_data")
        assert "findings" in result
        assert "impression" in result

    def test_analyze_pathology_slide(self) -> None:
        result = self.analyzer.analyze_pathology_slide(b"fake_slide_data")
        assert "findings" in result
        assert "grade" in result

    def test_segment_organs(self) -> None:
        result = self.analyzer.segment_organs(b"fake_ct_data")
        assert "segments" in result
        assert "liver" in result["segments"]
