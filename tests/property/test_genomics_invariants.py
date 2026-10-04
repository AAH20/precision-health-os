"""Property-based invariant tests for genomics module."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from precision_health_os.genomics import (
    GenotypeCall,
    PharmacogenomicGuideline,
    PharmacogenomicScorer,
    RiskPredictor,
    VariantCaller,
)
from precision_health_os.models import GenomicVariant

read_strategy = st.fixed_dictionaries(
    {
        "chrom": st.text(min_size=1, max_size=5),
        "pos": st.integers(min_value=0, max_value=1_000_000),
        "ref": st.sampled_from(["A", "C", "G", "T"]),
        "base": st.sampled_from(["A", "C", "G", "T"]),
        "quality": st.floats(min_value=0, max_value=100),
    }
)

variant_strategy = st.builds(
    GenomicVariant,
    chrom=st.text(min_size=1, max_size=5),
    pos=st.integers(min_value=1, max_value=1_000_000),
    ref=st.sampled_from(["A", "C", "G", "T"]),
    alt=st.sampled_from(["A", "C", "G", "T"]),
    gene=st.one_of(st.none(), st.text(min_size=1, max_size=10)),
    clinical_significance=st.one_of(
        st.none(),
        st.sampled_from(["pathogenic", "likely_pathogenic", "benign", "uncertain"]),
    ),
    zygosity=st.sampled_from(["homozygous", "heterozygous", "hemizygous"]),
)


class TestPharmacogenomicScorerInvariants:
    @given(
        gene=st.text(min_size=1, max_size=20),
        genotype=st.text(min_size=1, max_size=20),
        drug=st.text(min_size=1, max_size=20),
    )
    @settings(deadline=None, max_examples=30)
    def test_score_drug_returns_guideline_or_none(self, gene, genotype, drug):
        scorer = PharmacogenomicScorer()
        result = scorer.score_drug(gene, genotype, drug)
        assert result is None or isinstance(result, PharmacogenomicGuideline)


class TestVariantCallerInvariants:
    @given(reads=st.lists(read_strategy, min_size=0, max_size=50))
    @settings(deadline=None, max_examples=30)
    def test_call_variants_returns_list(self, reads):
        caller = VariantCaller()
        result = caller.call_variants(reads)
        assert isinstance(result, list)
        for call in result:
            assert isinstance(call, GenotypeCall)


class TestRiskPredictorInvariants:
    @given(variants=st.lists(variant_strategy, min_size=0, max_size=30))
    @settings(deadline=None, max_examples=30)
    def test_calculate_prs_returns_float(self, variants):
        predictor = RiskPredictor()
        predictor.train(variants, [0] * len(variants))
        result = predictor.calculate_prs(variants)
        assert isinstance(result, float)
        assert result >= 0.0
