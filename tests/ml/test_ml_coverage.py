"""Tests for precision_health_os.ml — covers uncovered branch 147->136."""

from precision_health_os.ml import DrugResponseModel


def test_predict_response_with_unrecognized_string_biomarker_value():
    """Cover branch 147->136: biomarker value is a str not in positive/negative lists.

    When a biomarker is in _biomarker_weights but its value is a string
    like "unknown" (not int/float, not positive, not negative), the elif
    at line 147 evaluates False and control returns to the for-loop at 136.
    """
    model = DrugResponseModel()
    result = model.predict_response(
        patient_id="P001",
        drug="trastuzumab",
        biomarkers={"HER2": "unknown"},
    )
    assert result.patient_id == "P001"
    assert result.drug_name == "trastuzumab"
    assert result.biomarkers == {"HER2": "unknown"}
    # has_matching_biomarker is True (HER2 is known), so we don't hit
    # the early-return at line 151. Score stays 0.0 → sigmoid(0)=0.5
    # → response_score 0.5 → not >0.6, but >0.4 → "intermediate"
    assert result.predicted_response == "intermediate"
