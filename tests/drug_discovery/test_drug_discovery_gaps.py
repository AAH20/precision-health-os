"""Gap-filling tests for drug_discovery module to raise coverage above 95%."""

from __future__ import annotations

import pytest

from precision_health_os.drug_discovery import (
    ADMETPredictor,
    ClinicalTrialMatcher,
    MolecularDocking,
    Molecule,
)


class TestADMETLipinskiViolations:
    """Test Lipinski violation accumulation for 0, 1, 2, 3, 4 violations."""

    def setup_method(self) -> None:
        self.predictor = ADMETPredictor()

    def test_zero_violations(self) -> None:
        mol = Molecule(smiles="CCO", name="ok", molecular_weight=300, logp=2.0, hbd=2, hba=5)
        result = self.predictor.predict(mol)
        assert result.absorption == 1.0
        assert result.bioavailability == 1.0
        assert result.toxicity == 0.3

    def test_one_violation_mw(self) -> None:
        mol = Molecule(smiles="C", name="big", molecular_weight=501, logp=2.0, hbd=2, hba=5)
        result = self.predictor.predict(mol)
        assert result.absorption == 0.8
        assert result.bioavailability == 0.75
        assert result.toxicity == 0.3

    def test_two_violations_mw_logp(self) -> None:
        mol = Molecule(smiles="C", name="big2", molecular_weight=501, logp=5.1, hbd=2, hba=5)
        result = self.predictor.predict(mol)
        assert result.absorption == 0.6
        assert result.bioavailability == 0.5
        assert result.toxicity == 0.7

    def test_three_violations_mw_logp_hbd(self) -> None:
        mol = Molecule(smiles="C", name="big3", molecular_weight=501, logp=5.1, hbd=6, hba=5)
        result = self.predictor.predict(mol)
        assert result.absorption == pytest.approx(0.4)
        assert result.bioavailability == pytest.approx(0.25)
        assert result.toxicity == 0.7

    def test_four_violations_all(self) -> None:
        mol = Molecule(smiles="C", name="big4", molecular_weight=501, logp=5.1, hbd=6, hba=11)
        result = self.predictor.predict(mol)
        assert result.absorption == pytest.approx(0.2)
        assert result.bioavailability == 0.0
        assert result.toxicity == 0.7


class TestADMETBoundaries:
    """Boundary tests for herg_risk and bbb_penetration."""

    def setup_method(self) -> None:
        self.predictor = ADMETPredictor()

    def test_herg_risk_false_at_boundary(self) -> None:
        mol = Molecule(smiles="C", name="b1", molecular_weight=400, logp=4.0)
        assert not self.predictor.predict(mol).herg_risk

    def test_herg_risk_true_above_boundary(self) -> None:
        mol = Molecule(smiles="C", name="b2", molecular_weight=401, logp=4.1)
        assert self.predictor.predict(mol).herg_risk

    def test_herg_risk_false_low_mw_high_logp(self) -> None:
        mol = Molecule(smiles="C", name="b3", molecular_weight=399, logp=5.0)
        assert not self.predictor.predict(mol).herg_risk

    def test_bbb_false_at_logp_boundary(self) -> None:
        mol = Molecule(smiles="C", name="b4", logp=2.0, tpsa=50)
        assert not self.predictor.predict(mol).bbb_penetration

    def test_bbb_true_above_logp_boundary(self) -> None:
        mol = Molecule(smiles="C", name="b5", logp=2.1, tpsa=50)
        assert self.predictor.predict(mol).bbb_penetration

    def test_bbb_false_at_tpsa_boundary(self) -> None:
        mol = Molecule(smiles="C", name="b6", logp=3.0, tpsa=90)
        assert not self.predictor.predict(mol).bbb_penetration

    def test_bbb_true_below_tpsa_boundary(self) -> None:
        mol = Molecule(smiles="C", name="b7", logp=3.0, tpsa=89)
        assert self.predictor.predict(mol).bbb_penetration


class TestClinicalTrialMatcherBranches:
    """Cover every branch in _calculate_match_score and the criteria helpers."""

    def setup_method(self) -> None:
        self.matcher = ClinicalTrialMatcher()

    def _patient(self, **kw: object) -> dict[str, object]:
        base: dict[str, object] = {
            "id": "p1",
            "age": 50,
            "conditions": ["lung_cancer"],
            "biomarkers": {"EGFR": "mutated"},
            "medications": ["cisplatin"],
        }
        base.update(kw)
        return base

    # --- _calculate_match_score branches ---

    def test_no_age_criteria_skips_age_check(self) -> None:
        """Covers 197->203: trial without min_age/max_age."""
        patient = self._patient()
        trials = [
            {"id": "t1", "name": "n", "eligibility_criteria": {"conditions": ["lung_cancer"]}}
        ]
        matches = self.matcher.match(patient, trials)  # type: ignore[arg-type]
        assert len(matches) == 1

    def test_no_condition_criteria_skips_condition_check(self) -> None:
        """Covers 207->211: trial with empty conditions list."""
        patient = self._patient()
        trials = [
            {
                "id": "t1",
                "name": "n",
                "eligibility_criteria": {"min_age": 18, "max_age": 75, "conditions": []},
            }
        ]
        matches = self.matcher.match(patient, trials)  # type: ignore[arg-type]
        assert len(matches) == 1

    def test_no_biomarker_match(self) -> None:
        """Covers 213->217: trial requires biomarkers patient doesn't have."""
        patient = self._patient(biomarkers={})
        trials = [
            {
                "id": "t1",
                "name": "n",
                "eligibility_criteria": {
                    "min_age": 18,
                    "max_age": 75,
                    "conditions": ["lung_cancer"],
                    "required_biomarkers": ["ALK"],
                },
            }
        ]
        matches = self.matcher.match(patient, trials)  # type: ignore[arg-type]
        assert len(matches) == 1

    def test_prior_treatments_match(self) -> None:
        """Covers 218-220: prior_treatments in criteria, patient has matching medication."""
        patient = self._patient()
        trials = [
            {
                "id": "t1",
                "name": "n",
                "eligibility_criteria": {
                    "min_age": 18,
                    "max_age": 75,
                    "conditions": ["lung_cancer"],
                    "prior_treatments": ["cisplatin"],
                },
            }
        ]
        matches = self.matcher.match(patient, trials)  # type: ignore[arg-type]
        assert len(matches) == 1
        assert matches[0]["score"] > 0.6  # 0.2 age + 0.4 conditions + 0.1 prior

    def test_prior_treatments_no_match(self) -> None:
        """Covers 218-220 else: prior_treatments in criteria, no matching medication."""
        patient = self._patient(medications=["aspirin"])
        trials = [
            {
                "id": "t1",
                "name": "n",
                "eligibility_criteria": {
                    "min_age": 18,
                    "max_age": 75,
                    "conditions": ["lung_cancer"],
                    "prior_treatments": ["cisplatin"],
                },
            }
        ]
        matches = self.matcher.match(patient, trials)  # type: ignore[arg-type]
        assert len(matches) == 1
        assert matches[0]["score"] < 0.7  # 0.2 age + 0.4 conditions, no +0.1

    # --- _get_matching_criteria branches ---

    def test_matching_criteria_age_fail(self) -> None:
        """Covers 230->237: age outside range in _get_matching_criteria."""
        patient = self._patient(age=80)
        trials = [
            {
                "id": "t1",
                "name": "n",
                "eligibility_criteria": {
                    "min_age": 18,
                    "max_age": 75,
                    "conditions": ["lung_cancer"],
                },
            }
        ]
        # Force _get_matching_criteria by calling match (which calls it when score > 0)
        # But age fails so score = 0 and match returns empty.
        # Call _get_matching_criteria directly:
        result = self.matcher._get_matching_criteria(patient, trials[0])  # type: ignore[arg-type]
        assert "age" not in result

    def test_matching_criteria_conditions_fail(self) -> None:
        """Covers 239->242: conditions don't match in _get_matching_criteria."""
        patient = self._patient(conditions=["diabetes"])
        trials = [
            {
                "id": "t1",
                "name": "n",
                "eligibility_criteria": {
                    "min_age": 18,
                    "max_age": 75,
                    "conditions": ["lung_cancer"],
                },
            }
        ]
        result = self.matcher._get_matching_criteria(patient, trials[0])  # type: ignore[arg-type]
        assert "conditions" not in result

    # --- _get_failing_criteria branches ---

    def test_failing_criteria_age(self) -> None:
        """Covers 255: age outside range in _get_failing_criteria."""
        patient = self._patient(age=80)
        trials = [
            {
                "id": "t1",
                "name": "n",
                "eligibility_criteria": {"min_age": 18, "max_age": 75},
            }
        ]
        result = self.matcher._get_failing_criteria(patient, trials[0])  # type: ignore[arg-type]
        assert "age" in result

    def test_failing_criteria_age_pass(self) -> None:
        """Age in range: no failing criteria."""
        patient = self._patient(age=50)
        trials = [
            {
                "id": "t1",
                "name": "n",
                "eligibility_criteria": {"min_age": 18, "max_age": 75},
            }
        ]
        result = self.matcher._get_failing_criteria(patient, trials[0])  # type: ignore[arg-type]
        assert "age" not in result


class TestVirtualScreenOrdering:
    """Verify virtual_screen returns results sorted by binding_energy."""

    def setup_method(self) -> None:
        self.docking = MolecularDocking(receptor_pdb="r.pdb")

    def test_virtual_screen_sorted_by_energy(self) -> None:
        library = [
            Molecule(smiles="C", name="a", molecular_weight=100, logp=1.0),
            Molecule(smiles="C", name="b", molecular_weight=500, logp=1.0),
            Molecule(smiles="C", name="c", molecular_weight=200, logp=1.0),
            Molecule(smiles="C", name="d", molecular_weight=400, logp=1.0),
        ]
        results = self.docking.virtual_screen(library, top_n=4)
        assert len(results) == 4
        for i in range(len(results) - 1):
            assert results[i].binding_energy <= results[i + 1].binding_energy

    def test_virtual_screen_top_n_limits_results(self) -> None:
        library = [
            Molecule(smiles="C", name=f"m{i}", molecular_weight=100 + i * 10, logp=1.0)
            for i in range(10)
        ]
        results = self.docking.virtual_screen(library, top_n=3)
        assert len(results) == 3
