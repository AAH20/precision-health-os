"""Coverage-gap and boundary tests for the ML module.

These are characterization tests for existing behavior: they pin down
contributing-factor thresholds, drug-response classification boundaries,
and confidence-scoring consistency.
"""

from datetime import datetime

import pytest

from precision_health_os.ml import DiseasePredictor, DrugResponseModel, MedicalImageAnalyzer
from precision_health_os.models import Patient


def _patient() -> Patient:
    return Patient(
        id="p001",
        mrn="MRN001",
        name="Test Patient",
        date_of_birth=datetime(1970, 1, 1),
        sex="male",
    )


class TestDiseasePredictorBoundaries:
    """Each contributing-factor threshold is tested at and around its boundary."""

    def setup_method(self) -> None:
        self.predictor = DiseasePredictor()
        self.patient = _patient()

    # --- Cardiovascular thresholds ---

    def test_cv_systolic_bp_triggers_above_140(self) -> None:
        r = self.predictor.predict_cardiovascular_risk(
            self.patient,
            {"systolic_bp": 141, "total_cholesterol": 200, "hdl": 50, "glucose": 100, "bmi": 25},
        )
        assert r.contributing_factors["hypertension"] == 0.2

    def test_cv_systolic_bp_no_trigger_at_140(self) -> None:
        r = self.predictor.predict_cardiovascular_risk(
            self.patient,
            {"systolic_bp": 140, "total_cholesterol": 200, "hdl": 50, "glucose": 100, "bmi": 25},
        )
        assert "hypertension" not in r.contributing_factors

    def test_cv_cholesterol_triggers_above_240(self) -> None:
        r = self.predictor.predict_cardiovascular_risk(
            self.patient,
            {"systolic_bp": 120, "total_cholesterol": 241, "hdl": 50, "glucose": 100, "bmi": 25},
        )
        assert r.contributing_factors["hyperlipidemia"] == 0.15

    def test_cv_cholesterol_no_trigger_at_240(self) -> None:
        r = self.predictor.predict_cardiovascular_risk(
            self.patient,
            {"systolic_bp": 120, "total_cholesterol": 240, "hdl": 50, "glucose": 100, "bmi": 25},
        )
        assert "hyperlipidemia" not in r.contributing_factors

    def test_cv_hdl_triggers_below_40(self) -> None:
        r = self.predictor.predict_cardiovascular_risk(
            self.patient,
            {"systolic_bp": 120, "total_cholesterol": 200, "hdl": 39, "glucose": 100, "bmi": 25},
        )
        assert r.contributing_factors["low_hdl"] == 0.1

    def test_cv_hdl_no_trigger_at_40(self) -> None:
        r = self.predictor.predict_cardiovascular_risk(
            self.patient,
            {"systolic_bp": 120, "total_cholesterol": 200, "hdl": 40, "glucose": 100, "bmi": 25},
        )
        assert "low_hdl" not in r.contributing_factors

    def test_cv_glucose_triggers_above_126(self) -> None:
        r = self.predictor.predict_cardiovascular_risk(
            self.patient,
            {"systolic_bp": 120, "total_cholesterol": 200, "hdl": 50, "glucose": 127, "bmi": 25},
        )
        assert r.contributing_factors["diabetes"] == 0.2

    def test_cv_glucose_no_trigger_at_126(self) -> None:
        r = self.predictor.predict_cardiovascular_risk(
            self.patient,
            {"systolic_bp": 120, "total_cholesterol": 200, "hdl": 50, "glucose": 126, "bmi": 25},
        )
        assert "diabetes" not in r.contributing_factors

    def test_cv_bmi_triggers_above_30(self) -> None:
        r = self.predictor.predict_cardiovascular_risk(
            self.patient,
            {"systolic_bp": 120, "total_cholesterol": 200, "hdl": 50, "glucose": 100, "bmi": 31},
        )
        assert r.contributing_factors["obesity"] == 0.1

    def test_cv_bmi_no_trigger_at_30(self) -> None:
        r = self.predictor.predict_cardiovascular_risk(
            self.patient,
            {"systolic_bp": 120, "total_cholesterol": 200, "hdl": 50, "glucose": 100, "bmi": 30},
        )
        assert "obesity" not in r.contributing_factors

    # --- Diabetes thresholds ---

    def test_diabetes_glucose_triggers_at_126(self) -> None:
        r = self.predictor.predict_diabetes_risk(
            self.patient, {"fasting_glucose": 126, "hba1c": 5.0, "bmi": 22}
        )
        assert r.contributing_factors["fasting_glucose"] == 0.4

    def test_diabetes_glucose_impaired_at_100(self) -> None:
        r = self.predictor.predict_diabetes_risk(
            self.patient, {"fasting_glucose": 100, "hba1c": 5.0, "bmi": 22}
        )
        assert r.contributing_factors["impaired_fasting_glucose"] == 0.2

    def test_diabetes_glucose_no_trigger_at_99(self) -> None:
        r = self.predictor.predict_diabetes_risk(
            self.patient, {"fasting_glucose": 99, "hba1c": 5.0, "bmi": 22}
        )
        assert "impaired_fasting_glucose" not in r.contributing_factors
        assert "fasting_glucose" not in r.contributing_factors

    def test_diabetes_hba1c_triggers_at_6_5(self) -> None:
        r = self.predictor.predict_diabetes_risk(
            self.patient, {"fasting_glucose": 90, "hba1c": 6.5, "bmi": 22}
        )
        assert r.contributing_factors["hba1c"] == 0.4

    def test_diabetes_hba1c_prediabetes_at_5_7(self) -> None:
        r = self.predictor.predict_diabetes_risk(
            self.patient, {"fasting_glucose": 90, "hba1c": 5.7, "bmi": 22}
        )
        assert r.contributing_factors["prediabetes_hba1c"] == 0.2

    def test_diabetes_hba1c_no_trigger_at_5_6(self) -> None:
        r = self.predictor.predict_diabetes_risk(
            self.patient, {"fasting_glucose": 90, "hba1c": 5.6, "bmi": 22}
        )
        assert "prediabetes_hba1c" not in r.contributing_factors
        assert "hba1c" not in r.contributing_factors

    def test_diabetes_bmi_obesity_at_30(self) -> None:
        r = self.predictor.predict_diabetes_risk(
            self.patient, {"fasting_glucose": 90, "hba1c": 5.0, "bmi": 30}
        )
        assert r.contributing_factors["obesity"] == 0.15

    def test_diabetes_bmi_overweight_at_25(self) -> None:
        r = self.predictor.predict_diabetes_risk(
            self.patient, {"fasting_glucose": 90, "hba1c": 5.0, "bmi": 25}
        )
        assert r.contributing_factors["overweight"] == 0.05

    def test_diabetes_bmi_no_trigger_at_24(self) -> None:
        r = self.predictor.predict_diabetes_risk(
            self.patient, {"fasting_glucose": 90, "hba1c": 5.0, "bmi": 24}
        )
        assert "overweight" not in r.contributing_factors
        assert "obesity" not in r.contributing_factors


class TestDrugResponseModelGaps:
    """Numeric values, weight==0, no-match early return, intermediate, loop back-edge."""

    def setup_method(self) -> None:
        self.model = DrugResponseModel()

    def test_numeric_int_value(self) -> None:
        r = self.model.predict_response("p1", "trastuzumab", {"HER2": 1})
        assert r.predicted_response == "responder"
        assert "HER2=1" in r.evidence[0]

    def test_numeric_float_value(self) -> None:
        r = self.model.predict_response("p1", "trastuzumab", {"HER2": 0.5})
        assert r.predicted_response == "intermediate"

    def test_positive_string_high(self) -> None:
        r = self.model.predict_response("p1", "trastuzumab", {"HER2": "high"})
        assert r.predicted_response == "responder"

    def test_positive_string_mutated(self) -> None:
        r = self.model.predict_response("p1", "trastuzumab", {"HER2": "mutated"})
        assert r.predicted_response == "responder"

    def test_negative_string_low(self) -> None:
        r = self.model.predict_response("p1", "trastuzumab", {"HER2": "low"})
        assert r.predicted_response == "non-responder"

    def test_negative_string_wild_type(self) -> None:
        r = self.model.predict_response("p1", "trastuzumab", {"HER2": "wild-type"})
        assert r.predicted_response == "non-responder"

    def test_weight_zero_drug_not_mapped(self) -> None:
        # HER2 is in weights but gefitinib is not mapped -> weight 0, no match
        r = self.model.predict_response("p1", "gefitinib", {"HER2": "positive"})
        assert r.predicted_response == "non-responder"
        assert r.confidence == 0.3

    def test_no_matching_biomarker_early_return(self) -> None:
        r = self.model.predict_response("p1", "trastuzumab", {"UNKNOWN": "positive"})
        assert r.predicted_response == "non-responder"
        assert r.confidence == 0.3
        assert r.evidence == ["No matching biomarkers for this drug"]

    def test_loop_back_edge_non_matching_then_matching(self) -> None:
        r = self.model.predict_response(
            "p1", "trastuzumab", {"UNKNOWN": "positive", "HER2": "positive"}
        )
        assert r.predicted_response == "responder"

    def test_loop_back_edge_negative_then_other(self) -> None:
        # HER2 negative: score -= 0.8; UNKNOWN skipped. Covers 147->136.
        r = self.model.predict_response(
            "p1", "trastuzumab", {"HER2": "negative", "UNKNOWN": "positive"}
        )
        assert r.predicted_response == "non-responder"

    def test_negative_string_then_next_biomarker(self) -> None:
        # HER2 "negative" for gefitinib (weight 0) then EGFR "positive" (0.7).
        # Covers the 147->136 loop back-edge after a negative-string biomarker.
        r = self.model.predict_response("p1", "gefitinib", {"HER2": "negative", "EGFR": "positive"})
        assert r.predicted_response == "responder"

    def test_intermediate_classification(self) -> None:
        r = self.model.predict_response("p1", "trastuzumab", {"HER2": 0.25})
        assert r.predicted_response == "intermediate"


class TestDrugResponseConfidenceConsistency:
    """Responder/intermediate/non-responder boundaries and confidence consistency."""

    def setup_method(self) -> None:
        self.model = DrugResponseModel()

    def test_classification_monotonic_with_score(self) -> None:
        responses = []
        for v in [-2.0, -0.6, -0.3, 0.0, 0.3, 0.6, 2.0]:
            r = self.model.predict_response("p1", "trastuzumab", {"HER2": v})
            responses.append(r.predicted_response)
        assert responses == [
            "non-responder",
            "non-responder",
            "intermediate",
            "intermediate",
            "intermediate",
            "responder",
            "responder",
        ]

    def test_confidence_symmetric(self) -> None:
        r_pos = self.model.predict_response("p1", "trastuzumab", {"HER2": 1.0})
        r_neg = self.model.predict_response("p1", "trastuzumab", {"HER2": -1.0})
        assert r_pos.confidence == pytest.approx(r_neg.confidence)

    def test_confidence_increases_with_extremity(self) -> None:
        r_mid = self.model.predict_response("p1", "trastuzumab", {"HER2": 0.0})
        r_extreme = self.model.predict_response("p1", "trastuzumab", {"HER2": 5.0})
        assert r_extreme.confidence > r_mid.confidence

    def test_confidence_capped_at_0_95(self) -> None:
        r = self.model.predict_response("p1", "trastuzumab", {"HER2": 100.0})
        assert r.confidence == pytest.approx(0.95)


class TestMedicalImageAnalyzerDetailed:
    """All three analyzer methods with structural assertions."""

    def setup_method(self) -> None:
        self.analyzer = MedicalImageAnalyzer()

    def test_chest_xray_structure(self) -> None:
        r = self.analyzer.analyze_chest_xray(b"data")
        assert len(r["findings"]) == 2
        assert r["findings"][0]["label"] == "pneumonia"
        assert "impression" in r
        assert "recommendation" in r

    def test_pathology_slide_structure(self) -> None:
        r = self.analyzer.analyze_pathology_slide(b"data")
        assert r["findings"][0]["label"] == "malignant"
        assert r["grade"] == "G3"

    def test_segment_organs_structure(self) -> None:
        r = self.analyzer.segment_organs(b"data")
        assert "liver" in r["segments"]
        assert "kidney_left" in r["segments"]
        assert "kidney_right" in r["segments"]
