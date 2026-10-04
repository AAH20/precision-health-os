"""Tests for drug discovery module."""

from precision_health_os.drug_discovery import (
    ADMETPredictor,
    ClinicalTrialMatcher,
    MolecularDocking,
    Molecule,
)


class TestMolecularDocking:
    """Tests for MolecularDocking."""

    def setup_method(self) -> None:
        self.docking = MolecularDocking(receptor_pdb="test_receptor.pdb")

    def test_dock_molecule(self) -> None:
        ligand = Molecule(
            smiles="CCO",
            name="ethanol",
            molecular_weight=46.07,
            logp=-0.14,
            hbd=1,
            hba=1,
        )
        result = self.docking.dock(ligand, binding_site=[0.0, 0.0, 0.0])
        assert result.ligand.name == "ethanol"
        assert result.binding_energy < 0

    def test_virtual_screen(self) -> None:
        library = [
            Molecule(smiles="CCO", name=f"mol{i}", molecular_weight=40 + i, logp=1.0)
            for i in range(5)
        ]
        results = self.docking.virtual_screen(library, top_n=3)
        assert len(results) == 3
        # Results should be sorted by binding energy
        for i in range(len(results) - 1):
            assert results[i].binding_energy <= results[i + 1].binding_energy


class TestADMETPredictor:
    """Tests for ADMETPredictor."""

    def setup_method(self) -> None:
        self.predictor = ADMETPredictor()

    def test_drug_like_molecule(self) -> None:
        molecule = Molecule(
            smiles="CCO",
            name="good_drug",
            molecular_weight=300,
            logp=2.0,
            hbd=2,
            hba=5,
            tpsa=60,
        )
        result = self.predictor.predict(molecule)
        assert result.absorption > 0.5
        assert result.bioavailability > 0.5
        assert not result.herg_risk

    def test_poor_drug_molecule(self) -> None:
        molecule = Molecule(
            smiles="CCCCCCCCCCCCCCCCCCCC",
            name="bad_drug",
            molecular_weight=800,
            logp=8.0,
            hbd=10,
            hba=20,
            tpsa=200,
        )
        result = self.predictor.predict(molecule)
        assert result.absorption < 0.5
        assert result.bioavailability < 0.5

    def test_drug_likeness_score(self) -> None:
        good = Molecule(
            smiles="CCO",
            name="good",
            molecular_weight=300,
            logp=2.0,
            hbd=2,
            hba=5,
            tpsa=60,
        )
        bad = Molecule(
            smiles="CCCC",
            name="bad",
            molecular_weight=800,
            logp=8.0,
            hbd=10,
            hba=20,
            tpsa=200,
        )
        assert self.predictor.drug_likeness_score(good) > self.predictor.drug_likeness_score(bad)


class TestClinicalTrialMatcher:
    """Tests for ClinicalTrialMatcher."""

    def setup_method(self) -> None:
        self.matcher = ClinicalTrialMatcher()

    def test_perfect_match(self) -> None:
        patient = {
            "id": "p001",
            "age": 55,
            "conditions": ["lung_cancer"],
            "biomarkers": {"EGFR": "mutated"},
            "medications": [],
        }
        trials = [
            {
                "id": "t001",
                "name": "EGFR Inhibitor Study",
                "eligibility_criteria": {
                    "min_age": 18,
                    "max_age": 75,
                    "conditions": ["lung_cancer"],
                    "required_biomarkers": ["EGFR"],
                },
            }
        ]
        matches = self.matcher.match(patient, trials)
        assert len(matches) == 1
        assert matches[0]["score"] > 0.5

    def test_no_match(self) -> None:
        patient = {
            "id": "p001",
            "age": 55,
            "conditions": ["diabetes"],
            "biomarkers": {},
            "medications": [],
        }
        trials = [
            {
                "id": "t001",
                "name": "Cancer Study",
                "eligibility_criteria": {
                    "min_age": 18,
                    "max_age": 75,
                    "conditions": ["lung_cancer"],
                },
            }
        ]
        matches = self.matcher.match(patient, trials)
        assert len(matches) == 0

    def test_age_filter(self) -> None:
        patient = {
            "id": "p001",
            "age": 80,
            "conditions": ["lung_cancer"],
            "biomarkers": {},
            "medications": [],
        }
        trials = [
            {
                "id": "t001",
                "name": "Study",
                "eligibility_criteria": {
                    "min_age": 18,
                    "max_age": 75,
                    "conditions": ["lung_cancer"],
                },
            }
        ]
        matches = self.matcher.match(patient, trials)
        assert len(matches) == 0
