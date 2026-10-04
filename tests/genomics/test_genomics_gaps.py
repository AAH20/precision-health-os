"""Coverage-gap tests for precision_health_os.genomics.

Targets the previously uncovered branches:
- VariantCaller alt-fraction boundaries (exactly 0.2 / 0.8)
- PharmacogenomicScorer.calculate_phenotype_score intermediate/extensive paths
- RiskPredictor.train/calculate_prs homozygous vs heterozygous weighting
- RiskPredictor.risk_category exact z boundaries and zero-std guard
- PharmacogenomicScorer.get_all_recommendations no-match combinations
- Regression: genotype-aware scoring gives poor metabolizers non-standard advice
"""

import pytest

from precision_health_os.genomics import (
    PharmacogenomicScorer,
    RiskPredictor,
    VariantCaller,
)
from precision_health_os.models import GenomicVariant


def _variant(
    chrom="1", pos=100, ref="A", alt="G", gene=None, significance=None, zygosity="heterozygous"
):
    return GenomicVariant(
        chrom=chrom,
        pos=pos,
        ref=ref,
        alt=alt,
        gene=gene,
        clinical_significance=significance,
        zygosity=zygosity,
    )


class TestVariantCallerAltFractionBoundaries:
    """Genotype calls at the exact 0.2 / 0.8 alt-fraction boundaries."""

    @staticmethod
    def _reads(pos, ref, alt, n_ref, n_alt, chrom="1", quality=50):
        return [
            {"chrom": chrom, "pos": pos, "ref": ref, "base": ref, "quality": quality}
            for _ in range(n_ref)
        ] + [
            {"chrom": chrom, "pos": pos, "ref": ref, "base": alt, "quality": quality}
            for _ in range(n_alt)
        ]

    def test_alt_fraction_exactly_0_2_calls_homozygous_ref(self):
        # 8 ref + 2 alt of 10 reads -> alt_frac == 0.2 -> not > 0.2 -> 0/0, no call
        caller = VariantCaller()
        calls = caller.call_variants(self._reads(100, "A", "G", 8, 2))
        assert calls == []

    def test_alt_fraction_exactly_0_8_calls_heterozygous(self):
        # 2 ref + 8 alt of 10 reads -> alt_frac == 0.8 -> not > 0.8 -> 0/1
        caller = VariantCaller()
        calls = caller.call_variants(self._reads(100, "A", "G", 2, 8))
        assert len(calls) == 1
        assert calls[0].genotype == "0/1"
        assert calls[0].ref == "A"
        assert calls[0].alt == "G"
        assert calls[0].depth == 10
        assert calls[0].allele_depth == {"A": 2, "G": 8}

    def test_alt_fraction_above_0_8_calls_homozygous_alt(self):
        caller = VariantCaller()
        calls = caller.call_variants(self._reads(100, "A", "G", 1, 9))
        assert len(calls) == 1
        assert calls[0].genotype == "1/1"
        assert calls[0].allele_depth == {"A": 1, "G": 9}

    def test_alt_fraction_between_boundaries_calls_heterozygous(self):
        caller = VariantCaller()
        calls = caller.call_variants(self._reads(100, "A", "G", 5, 5))
        assert len(calls) == 1
        assert calls[0].genotype == "0/1"


class TestPhenotypeScoreCategories:
    """calculate_phenotype_score branch coverage (intermediate / extensive)."""

    def test_single_pathogenic_variant_is_intermediate(self):
        scorer = PharmacogenomicScorer()
        variants = [_variant(gene="CYP2D6", significance="pathogenic")]
        assert scorer.calculate_phenotype_score("CYP2D6", variants) == "intermediate_metabolizer"

    def test_single_likely_pathogenic_variant_is_intermediate(self):
        scorer = PharmacogenomicScorer()
        variants = [_variant(gene="CYP2D6", significance="likely_pathogenic")]
        assert scorer.calculate_phenotype_score("CYP2D6", variants) == "intermediate_metabolizer"

    def test_two_pathogenic_variants_is_poor_metabolizer(self):
        scorer = PharmacogenomicScorer()
        variants = [
            _variant(pos=100, gene="CYP2D6", significance="pathogenic"),
            _variant(pos=200, gene="CYP2D6", significance="likely_pathogenic"),
        ]
        assert scorer.calculate_phenotype_score("CYP2D6", variants) == "poor_metabolizer"

    def test_no_pathogenic_variants_in_gene_is_extensive(self):
        scorer = PharmacogenomicScorer()
        variants = [
            _variant(pos=100, gene="CYP2D6", significance="benign"),
            _variant(pos=200, gene="CYP2D6", significance="likely_benign"),
            _variant(pos=300, gene="TPMT", significance="pathogenic"),
        ]
        assert scorer.calculate_phenotype_score("CYP2D6", variants) == "extensive_metabolizer"

    def test_empty_variant_list_is_extensive(self):
        scorer = PharmacogenomicScorer()
        assert scorer.calculate_phenotype_score("CYP2D6", []) == "extensive_metabolizer"


class TestRiskPredictorTrainAndPrs:
    """train() weight seeding and calculate_prs() zygosity weighting."""

    def test_train_then_prs_homozygous_vs_heterozygous_weighting(self):
        predictor = RiskPredictor()
        predictor.train([_variant()], phenotypes=[0, 1])
        het = _variant(zygosity="heterozygous")
        hom = _variant(zygosity="homozygous")
        assert predictor.calculate_prs([het]) == pytest.approx(0.01)
        assert predictor.calculate_prs([hom]) == pytest.approx(0.02)
        assert predictor.calculate_prs([het, hom]) == pytest.approx(0.03)

    def test_prs_before_training_is_zero(self):
        predictor = RiskPredictor()
        assert predictor.calculate_prs([_variant(zygosity="homozygous")]) == 0.0

    def test_prs_unknown_variant_contributes_zero(self):
        predictor = RiskPredictor()
        predictor.train([_variant()], phenotypes=[1])
        unknown = _variant(chrom="2", pos=999, ref="C", alt="T")
        assert predictor.calculate_prs([unknown]) == 0.0


class TestRiskPredictorBoundaries:
    """risk_category at exact z boundaries and the zero-std guard."""

    def test_z_exactly_2_is_high_not_very_high(self):
        predictor = RiskPredictor()
        # z = (2.0 - 1.0) / 0.5 == 2.0 exactly
        assert predictor.risk_category(prs=2.0, population_mean=1.0, population_std=0.5) == "high"

    def test_z_exactly_1_is_average(self):
        predictor = RiskPredictor()
        # z = (1.5 - 1.0) / 0.5 == 1.0 exactly
        assert (
            predictor.risk_category(prs=1.5, population_mean=1.0, population_std=0.5) == "average"
        )

    def test_z_exactly_minus_1_is_average(self):
        predictor = RiskPredictor()
        # z = (0.5 - 1.0) / 0.5 == -1.0 exactly
        assert (
            predictor.risk_category(prs=0.5, population_mean=1.0, population_std=0.5) == "average"
        )

    def test_z_above_2_is_very_high(self):
        predictor = RiskPredictor()
        assert (
            predictor.risk_category(prs=2.5, population_mean=1.0, population_std=0.5) == "very_high"
        )

    def test_z_between_1_and_2_is_high(self):
        predictor = RiskPredictor()
        assert predictor.risk_category(prs=1.75, population_mean=1.0, population_std=0.5) == "high"

    def test_z_below_minus_1_is_low(self):
        predictor = RiskPredictor()
        assert predictor.risk_category(prs=0.0, population_mean=1.0, population_std=0.5) == "low"

    def test_zero_population_std_returns_average(self):
        predictor = RiskPredictor()
        assert (
            predictor.risk_category(prs=9.9, population_mean=1.0, population_std=0.0) == "average"
        )


class TestPharmacogenomicGenotypeAwareScoring:
    """Genotype-aware matching and no-match recommendation paths."""

    def test_poor_metabolizer_gets_non_standard_codeine_advice(self):
        # Regression: scoring must match genotype, not just gene+drug.
        scorer = PharmacogenomicScorer()
        rec = scorer.score_drug("CYP2D6", "*4/*4", "codeine")
        assert rec is not None
        assert rec.phenotype == "poor_metabolizer"
        assert rec.recommendation != "Standard dosing"
        assert "Avoid" in rec.recommendation

    def test_ultrarapid_metabolizer_gets_non_standard_codeine_advice(self):
        scorer = PharmacogenomicScorer()
        rec = scorer.score_drug("CYP2D6", "*1xN/*1", "codeine")
        assert rec is not None
        assert rec.phenotype == "ultrarapid_metabolizer"
        assert "Avoid" in rec.recommendation

    def test_extensive_metabolizer_gets_standard_dosing(self):
        scorer = PharmacogenomicScorer()
        rec = scorer.score_drug("CYP2D6", "*1/*1", "codeine")
        assert rec is not None
        assert rec.phenotype == "extensive_metabolizer"
        assert rec.recommendation == "Standard dosing"

    def test_drug_name_matching_is_case_insensitive(self):
        scorer = PharmacogenomicScorer()
        rec = scorer.score_drug("CYP2D6", "*1/*1", "Codeine")
        assert rec is not None
        assert rec.recommendation == "Standard dosing"

    def test_unlisted_genotype_returns_none(self):
        scorer = PharmacogenomicScorer()
        assert scorer.score_drug("CYP2D6", "*2/*2", "codeine") is None

    def test_get_all_recommendations_gene_without_drug_match_returns_empty(self):
        scorer = PharmacogenomicScorer()
        # CYP2D6 has no clopidogrel guideline
        assert scorer.get_all_recommendations({"CYP2D6": "*1/*1"}, ["clopidogrel"]) == []

    def test_get_all_recommendations_unknown_gene_returns_empty(self):
        scorer = PharmacogenomicScorer()
        assert scorer.get_all_recommendations({"CYP3A5": "*1/*1"}, ["codeine"]) == []

    def test_get_all_recommendations_returns_only_matching_combinations(self):
        scorer = PharmacogenomicScorer()
        recs = scorer.get_all_recommendations(
            {"CYP2D6": "*1/*1", "CYP2C19": "*1/*1"}, ["codeine", "clopidogrel"]
        )
        assert len(recs) == 2
        assert {r.gene for r in recs} == {"CYP2D6", "CYP2C19"}
        assert {r.drug for r in recs} == {"codeine", "clopidogrel"}
