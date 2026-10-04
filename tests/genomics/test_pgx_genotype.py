"""Regression tests for genotype-aware pharmacogenomic scoring.

Regression: score_drug() ignored its `genotype` argument and returned the
first guideline matching the drug, so a poor metabolizer (e.g. CYP2D6 *4/*4)
was told "Standard dosing" for codeine — a clinically unsafe recommendation.
"""

from precision_health_os.genomics import PharmacogenomicScorer


class TestGenotypeAwareScoring:
    """score_drug must key on genotype, not just gene+drug."""

    def setup_method(self) -> None:
        self.scorer = PharmacogenomicScorer()

    def test_poor_metabolizer_gets_nonstandard_advice(self) -> None:
        """CYP2D6 *4/*4 is a poor metabolizer — must NOT get standard dosing."""
        rec = self.scorer.score_drug("CYP2D6", "*4/*4", "codeine")
        assert rec is not None
        assert rec.phenotype == "poor_metabolizer"
        assert rec.allele == "*4/*4"
        assert "standard" not in rec.recommendation.lower()

    def test_ultrarapid_metabolizer_gets_nonstandard_advice(self) -> None:
        """CYP2D6 *1xN/*1 is an ultrarapid metabolizer — toxicity risk."""
        rec = self.scorer.score_drug("CYP2D6", "*1xN/*1", "codeine")
        assert rec is not None
        assert rec.phenotype == "ultrarapid_metabolizer"
        assert "standard" not in rec.recommendation.lower()

    def test_normal_genotype_gets_standard_dosing(self) -> None:
        """CYP2D6 *1/*1 is an extensive metabolizer — standard dosing."""
        rec = self.scorer.score_drug("CYP2D6", "*1/*1", "codeine")
        assert rec is not None
        assert rec.phenotype == "extensive_metabolizer"
        assert "standard" in rec.recommendation.lower()

    def test_genotype_selects_distinct_guidelines(self) -> None:
        """Different genotypes for the same drug must yield different advice."""
        normal = self.scorer.score_drug("CYP2D6", "*1/*1", "codeine")
        poor = self.scorer.score_drug("CYP2D6", "*4/*4", "codeine")
        assert normal is not None and poor is not None
        assert normal.recommendation != poor.recommendation
        assert normal.phenotype != poor.phenotype

    def test_unknown_genotype_returns_none(self) -> None:
        """A genotype absent from the guideline table must not silently match."""
        assert self.scorer.score_drug("CYP2D6", "*99/*99", "codeine") is None

    def test_unknown_drug_returns_none(self) -> None:
        assert self.scorer.score_drug("CYP2D6", "*1/*1", "aspirin") is None

    def test_all_documented_genes_resolve_their_own_genotypes(self) -> None:
        """Every curated (gene, genotype, drug) row must be retrievable."""
        for gene, guidelines in self.scorer.CPIC_GUIDELINES.items():
            for g in guidelines:
                got = self.scorer.score_drug(gene, g.allele, g.drug)
                assert got is not None, f"{gene} {g.allele} {g.drug} not retrievable"
                assert got.phenotype == g.phenotype
                assert got.recommendation == g.recommendation

    def test_get_all_recommendations_is_genotype_specific(self) -> None:
        """Batch lookup must respect genotype per gene."""
        recs = self.scorer.get_all_recommendations(
            {"CYP2D6": "*4/*4", "CYP2C19": "*2/*2"},
            ["codeine", "clopidogrel"],
        )
        phenotypes = {r.phenotype for r in recs}
        assert "poor_metabolizer" in phenotypes
        assert len(recs) == 2
        assert all(r.phenotype == "poor_metabolizer" for r in recs)
