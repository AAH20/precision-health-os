"""ML module: Disease prediction, drug response modeling, medical image analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from precision_health_os.models import DrugResponsePrediction, Patient
from precision_health_os.utils import normalize

logger = logging.getLogger(__name__)


@dataclass
class DiseaseRisk:
    """Disease risk prediction result."""

    disease: str
    risk_score: float  # 0-1
    confidence: float  # 0-1
    contributing_factors: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


class DiseasePredictor:
    """Disease risk prediction using ensemble methods.

    Production would use XGBoost, LightGBM, or neural networks.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._feature_importance: dict[str, dict[str, float]] = {}

    def predict_cardiovascular_risk(self, patient: Patient, vitals: dict[str, float]) -> DiseaseRisk:
        """Predict cardiovascular disease risk (simplified Framingham-like)."""
        age = 50  # Would compute from DOB
        contributing: dict[str, float] = {}

        # Simplified risk factors
        if vitals.get("systolic_bp", 120) > 140:
            contributing["hypertension"] = 0.2
        if vitals.get("total_cholesterol", 200) > 240:
            contributing["hyperlipidemia"] = 0.15
        if vitals.get("hdl", 50) < 40:
            contributing["low_hdl"] = 0.1
        if vitals.get("glucose", 100) > 126:
            contributing["diabetes"] = 0.2
        if vitals.get("bmi", 25) > 30:
            contributing["obesity"] = 0.1

        base_risk = 0.05
        risk = min(base_risk + sum(contributing.values()), 0.95)

        return DiseaseRisk(
            disease="cardiovascular_disease",
            risk_score=risk,
            confidence=0.75,
            contributing_factors=contributing,
            recommendations=[
                "Lifestyle modification",
                "Regular BP monitoring",
                "Lipid panel follow-up",
            ]
            if risk > 0.3
            else ["Continue current management"],
        )

    def predict_diabetes_risk(self, patient: Patient, labs: dict[str, float]) -> DiseaseRisk:
        """Predict type 2 diabetes risk."""
        contributing: dict[str, float] = {}

        glucose = labs.get("fasting_glucose", 100)
        hba1c = labs.get("hba1c", 5.5)
        bmi = labs.get("bmi", 25)

        if glucose >= 126:
            contributing["fasting_glucose"] = 0.4
        elif glucose >= 100:
            contributing["impaired_fasting_glucose"] = 0.2

        if hba1c >= 6.5:
            contributing["hba1c"] = 0.4
        elif hba1c >= 5.7:
            contributing["prediabetes_hba1c"] = 0.2

        if bmi >= 30:
            contributing["obesity"] = 0.15
        elif bmi >= 25:
            contributing["overweight"] = 0.05

        risk = min(sum(contributing.values()), 0.95)

        return DiseaseRisk(
            disease="type_2_diabetes",
            risk_score=risk,
            confidence=0.8,
            contributing_factors=contributing,
            recommendations=[
                "Weight management",
                "Dietary counseling",
                "Exercise program",
            ]
            if risk > 0.3
            else ["Annual screening"],
        )


class DrugResponseModel:
    """Drug response prediction using pharmacogenomic and clinical features."""

    def __init__(self) -> None:
        self._biomarker_weights: dict[str, dict[str, float]] = {
            "HER2": {"trastuzumab": 0.8},
            "EGFR": {"gefitinib": 0.7, "erlotinib": 0.7},
            "KRAS": {"cetuximab": -0.6},
            "BRAF": {"vemurafenib": 0.8},
            "PD-L1": {"pembrolizumab": 0.6},
        }

    def predict_response(
        self,
        patient_id: str,
        drug: str,
        biomarkers: dict[str, Any],
        clinical_features: dict[str, Any] | None = None,
    ) -> DrugResponsePrediction:
        """Predict drug response based on biomarkers."""
        score = 0.0
        evidence: list[str] = []
        has_matching_biomarker = False

        for biomarker, value in biomarkers.items():
            if biomarker in self._biomarker_weights:
                weight = self._biomarker_weights[biomarker].get(drug, 0)
                if weight != 0:
                    has_matching_biomarker = True
                if isinstance(value, (int, float)):
                    score += weight * value
                    evidence.append(f"{biomarker}={value} (weight: {weight})")
                elif isinstance(value, str) and value.lower() in ("positive", "high", "mutated"):
                    score += weight
                    evidence.append(f"{biomarker} {value} (weight: {weight})")
                elif isinstance(value, str) and value.lower() in ("negative", "low", "wild-type"):
                    score -= abs(weight)  # Negative biomarker reduces score
                    evidence.append(f"{biomarker} {value} (weight: -{abs(weight)})")

        if not has_matching_biomarker:
            return DrugResponsePrediction(
                patient_id=patient_id,
                drug_name=drug,
                predicted_response="non-responder",
                confidence=0.3,
                biomarkers=biomarkers,
                evidence=["No matching biomarkers for this drug"],
            )

        # Normalize score to 0-1 using sigmoid
        response_score = 1 / (1 + pow(2.71828, -score))

        if response_score > 0.6:
            response = "responder"
        elif response_score > 0.4:
            response = "intermediate"
        else:
            response = "non-responder"

        # Confidence: how far from 0.5 (uncertainty)
        confidence = min(abs(response_score - 0.5) * 3.0, 0.95)

        return DrugResponsePrediction(
            patient_id=patient_id,
            drug_name=drug,
            predicted_response=response,
            confidence=confidence,
            biomarkers=biomarkers,
            evidence=evidence,
        )


class MedicalImageAnalyzer:
    """Medical image analysis using deep learning.

    Production would use MONAI, nnU-Net, or custom PyTorch models.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def analyze_chest_xray(self, image_data: bytes) -> dict[str, Any]:
        """Analyze chest X-ray for pathologies."""
        # Placeholder: production would use DenseNet/ResNet
        return {
            "findings": [
                {"label": "pneumonia", "confidence": 0.85, "bbox": [100, 150, 200, 200]},
                {"label": "cardiomegaly", "confidence": 0.72, "bbox": [50, 80, 300, 250]},
            ],
            "impression": "Bilateral infiltrates consistent with pneumonia",
            "recommendation": "Clinical correlation recommended",
        }

    def analyze_pathology_slide(self, image_data: bytes) -> dict[str, Any]:
        """Analyze pathology slide for cancer detection."""
        return {
            "findings": [
                {"label": "malignant", "confidence": 0.92, "region": "top_left"},
            ],
            "grade": "G3",
            "recommendation": "Oncology referral",
        }

    def segment_organs(self, ct_data: bytes) -> dict[str, Any]:
        """Segment organs from CT scan."""
        return {
            "segments": {
                "liver": {"volume_ml": 1500, "mean_hu": 55},
                "kidney_left": {"volume_ml": 150, "mean_hu": 35},
                "kidney_right": {"volume_ml": 145, "mean_hu": 33},
            },
        }
