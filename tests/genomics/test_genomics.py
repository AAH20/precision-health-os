"""Tests for genomics module."""

from precision_health_os.genomics import (
    PharmacogenomicScorer,
    RiskPredictor,
    VariantCaller,
)
from precision_health_os.models import GenomicVariant


class TestVariantCaller:
    """Tests for VariantCaller."""

    def setup_method(self) -> None:
        self.caller = VariantCaller(min_quality=20, min_depth=5)

    def test_no_variants(self) -> None:
        reads = [
            {"chrom": "1", "pos": 100, "ref": "A", "base": "A", "quality": 40} for _ in range(10)
        ]
        calls = self.caller.call_variants(reads)
        assert len(calls) == 0

    def test_snp_call(self) -> None:
        reads = [
            {"chrom": "1", "pos": 100, "ref": "A", "base": "A", "quality": 40} for _ in range(5)
        ] + [{"chrom": "1", "pos": 100, "ref": "A", "base": "G", "quality": 40} for _ in range(5)]
        calls = self.caller.call_variants(reads)
        assert len(calls) == 1
        assert calls[0].alt == "G"
        assert calls[0].genotype == "0/1"

    def test_homozygous_alt(self) -> None:
        reads = [
            {"chrom": "1", "pos": 100, "ref": "A", "base": "G", "quality": 40} for _ in range(10)
        ]
        calls = self.caller.call_variants(reads)
        assert len(calls) == 1
        assert calls[0].genotype == "1/1"

    def test_low_quality_filtered(self) -> None:
        reads = [
            {"chrom": "1", "pos": 100, "ref": "A", "base": "G", "quality": 10} for _ in range(10)
        ]
        calls = self.caller.call_variants(reads)
        assert len(calls) == 0

    def test_low_depth_filtered(self) -> None:
        reads = [
            {"chrom": "1", "pos": 100, "ref": "A", "base": "G", "quality": 40} for _ in range(3)
        ]
        calls = self.caller.call_variants(reads)
        assert len(calls) == 0


class TestPharmacogenomicScorer:
    """Tests for PharmacogenomicScorer."""

    def setup_method(self) -> None:
        self.scorer = PharmacogenomicScorer()

    def test_score_drug_match(self) -> None:
        result = self.scorer.score_drug("CYP2D6", "*1/*1", "codeine")
        assert result is not None
        assert result.drug == "codeine"

    def test_score_drug_no_match(self) -> None:
        result = self.scorer.score_drug("CYP2D6", "*1/*1", "aspirin")
        assert result is None

    def test_get_all_recommendations(self) -> None:
        genotypes = {"CYP2D6": "*1/*1", "CYP2C19": "*2/*2"}
        drugs = ["codeine", "clopidogrel"]
        recs = self.scorer.get_all_recommendations(genotypes, drugs)
        assert len(recs) == 2

    def test_calculate_phenotype_score(self) -> None:
        variants = [
            GenomicVariant(
                chrom="1",
                pos=100,
                ref="A",
                alt="G",
                gene="CYP2D6",
                clinical_significance="pathogenic",
            ),
            GenomicVariant(
                chrom="1",
                pos=200,
                ref="C",
                alt="T",
                gene="CYP2D6",
                clinical_significance="pathogenic",
            ),
        ]
        score = self.scorer.calculate_phenotype_score("CYP2D6", variants)
        assert score == "poor_metabolizer"


class TestRiskPredictor:
    """Tests for RiskPredictor."""

    def setup_method(self) -> None:
        self.predictor = RiskPredictor()

    def test_calculate_prs(self) -> None:
        variants = [
            GenomicVariant(chrom="1", pos=100, ref="A", alt="G", zygosity="heterozygous"),
            GenomicVariant(chrom="2", pos=200, ref="C", alt="T", zygosity="homozygous"),
        ]
        self.predictor.train(variants, [1, 0, 1, 0])
        prs = self.predictor.calculate_prs(variants)
        assert prs >= 0

    def test_risk_category(self) -> None:
        assert self.predictor.risk_category(3.0, 0.0, 1.0) == "very_high"
        assert self.predictor.risk_category(1.5, 0.0, 1.0) == "high"
        assert self.predictor.risk_category(0.0, 0.0, 1.0) == "average"
        assert self.predictor.risk_category(-1.5, 0.0, 1.0) == "low"
