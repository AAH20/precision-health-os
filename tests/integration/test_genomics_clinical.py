"""Integration tests: genomics scoring → pharmacogenomic recommendation → clinical decision."""

from __future__ import annotations

from datetime import UTC, datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.clinical import ClinicalRule
from precision_health_os.genomics import PharmacogenomicScorer
from precision_health_os.models import AlertSeverity, Patient


class TestPharmacogenomicScoring:
    """Test pharmacogenomic scoring for drug-gene interactions."""

    def setup_method(self) -> None:
        self.scorer = PharmacogenomicScorer()

    def test_cyp2d6_poor_metabolizer_codeine_avoid(self) -> None:
        """CYP2D6 *4/*4 (poor metabolizer) on codeine → avoid codeine."""
        result = self.scorer.score_drug("CYP2D6", "*4/*4", "codeine")
        assert result is not None
        assert result.recommendation == "Avoid codeine — use alternative analgesic"
        assert result.phenotype == "poor_metabolizer"

    def test_cyp2d6_normal_metabolizer_codeine_standard_dosing(self) -> None:
        """CYP2D6 *1/*1 (extensive metabolizer) on codeine → standard dosing."""
        result = self.scorer.score_drug("CYP2D6", "*1/*1", "codeine")
        assert result is not None
        assert result.recommendation == "Standard dosing"
        assert result.phenotype == "extensive_metabolizer"

    def test_unknown_genotype_returns_none(self) -> None:
        """Unknown genotype → None returned."""
        result = self.scorer.score_drug("CYP2D6", "*99/*99", "codeine")
        assert result is None

    def test_recommendation_includes_evidence_level(self) -> None:
        """Recommendation includes evidence level."""
        result = self.scorer.score_drug("CYP2D6", "*4/*4", "codeine")
        assert result is not None
        assert result.evidence_level == "1A"

    def test_multiple_gene_drug_combinations(self) -> None:
        """Test multiple gene-drug combinations."""
        result = self.scorer.score_drug("CYP2C19", "*2/*2", "clopidogrel")
        assert result is not None
        assert result.recommendation == "Consider prasugrel or ticagrelor"

        result = self.scorer.score_drug("CYP2C9", "*3/*3", "warfarin")
        assert result is not None
        assert result.recommendation == "Reduce dose by 50-70%"

        result = self.scorer.score_drug("TPMT", "*2/*2", "azathioprine")
        assert result is not None
        assert result.recommendation == "Avoid or reduce dose by 90%"

        result = self.scorer.score_drug("SLCO1B1", "*5/*5", "simvastatin")
        assert result is not None
        assert result.recommendation == "Consider lower dose or alternative"


class TestGenomicsClinicalIntegration:
    """Integration tests: genomics → pharmacogenomic → clinical decision."""

    def setup_method(self) -> None:
        self.api = PrecisionHealthAPI()
        self.scorer = PharmacogenomicScorer()

    def test_full_flow_register_score_recommend(self) -> None:
        """Full flow: register patient → score drug → get recommendation."""
        patient = Patient(
            id="P001",
            mrn="MRN001",
            name="Test Patient",
            date_of_birth=datetime(1980, 1, 1, tzinfo=UTC),
            sex="male",
        )
        registered = self.api.register_patient(patient)
        assert registered.id == "P001"

        result = self.scorer.score_drug("CYP2D6", "*4/*4", "codeine")
        assert result is not None
        assert result.recommendation == "Avoid codeine — use alternative analgesic"

        fetched = self.api.get_patient("P001")
        assert fetched is not None
        assert fetched.name == "Test Patient"

    def test_full_flow_with_clinical_alert(self) -> None:
        """Full flow: register → score → generate clinical alert."""
        patient = Patient(
            id="P002",
            mrn="MRN002",
            name="Alert Patient",
            date_of_birth=datetime(1975, 6, 15, tzinfo=UTC),
            sex="female",
        )
        self.api.register_patient(patient)

        result = self.scorer.score_drug("CYP2D6", "*4/*4", "codeine")
        assert result is not None

        rule = ClinicalRule(
            id="PGX001",
            name="Poor Metabolizer Alert",
            condition="True",
            severity=AlertSeverity.HIGH,
            message="Patient is a poor metabolizer — avoid codeine",
            recommendations=["Use alternative analgesic"],
        )
        self.api.cds.add_rule(rule)

        alerts = self.api.cds.evaluate_rules(patient, {})
        assert len(alerts) >= 1
        assert any("poor metabolizer" in a.title.lower() for a in alerts)

    def test_get_all_recommendations(self) -> None:
        """Test get_all_recommendations with multiple genotypes and drugs."""
        genotypes = {"CYP2D6": "*4/*4", "CYP2C19": "*2/*2"}
        drugs = ["codeine", "clopidogrel"]
        recs = self.scorer.get_all_recommendations(genotypes, drugs)
        assert len(recs) == 2
        assert any(r.gene == "CYP2D6" for r in recs)
        assert any(r.gene == "CYP2C19" for r in recs)
