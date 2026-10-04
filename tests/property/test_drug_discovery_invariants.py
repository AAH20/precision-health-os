"""Property-based invariant tests for the drug_discovery module.

Invariants verified (for all generated inputs):
- MolecularDocking.dock always returns a well-formed DockingResult.
- ADMETPredictor.predict always returns an ADMETPrediction whose score
  fields are floats in [0, 1] and whose flag fields are booleans.
- ClinicalTrialMatcher.match always returns a list of scored trial dicts,
  sorted by descending score, each score in (0, 1].
- Every scoring function returns a float.
"""

from __future__ import annotations

import math
from dataclasses import asdict

from hypothesis import given, settings
from hypothesis import strategies as st

from precision_health_os.drug_discovery import (
    ADMETPrediction,
    ADMETPredictor,
    ClinicalTrialMatcher,
    DockingResult,
    MolecularDocking,
    Molecule,
)

invariant_settings = settings(max_examples=30, deadline=None)

finite_float = st.floats(allow_nan=False, allow_infinity=False)

molecule_strategy = st.builds(
    Molecule,
    smiles=st.text(min_size=1, max_size=20),
    name=st.text(max_size=20),
    molecular_weight=st.floats(
        min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
    ),
    logp=st.floats(min_value=-5.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    hbd=st.integers(min_value=0, max_value=20),
    hba=st.integers(min_value=0, max_value=20),
    tpsa=st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False),
    rotatable_bonds=st.integers(min_value=0, max_value=30),
)

# A binding site is either absent or a 3D coordinate (x, y, z).
binding_site_strategy = st.one_of(
    st.none(),
    st.lists(finite_float, min_size=3, max_size=3),
)

patient_strategy = st.fixed_dictionaries(
    {
        "age": st.integers(min_value=0, max_value=120),
        "conditions": st.lists(st.text(min_size=1, max_size=15), max_size=5),
        "biomarkers": st.dictionaries(st.text(min_size=1, max_size=10), st.booleans(), max_size=5),
        "medications": st.lists(st.text(min_size=1, max_size=15), max_size=5),
    }
)

trial_strategy = st.fixed_dictionaries(
    {
        "id": st.text(min_size=1, max_size=10),
        "name": st.text(max_size=30),
        "eligibility_criteria": st.fixed_dictionaries(
            {
                "min_age": st.integers(min_value=0, max_value=120),
                "max_age": st.integers(min_value=0, max_value=120),
                "conditions": st.lists(st.text(min_size=1, max_size=15), max_size=5),
                "required_biomarkers": st.lists(st.text(min_size=1, max_size=10), max_size=5),
                "prior_treatments": st.lists(st.text(min_size=1, max_size=15), max_size=5),
            }
        ),
    }
)

ADMET_SCORE_FIELDS = (
    "absorption",
    "distribution",
    "metabolism",
    "excretion",
    "toxicity",
    "bioavailability",
)
ADMET_FLAG_FIELDS = ("bbb_penetration", "herg_risk", "ames_mutagenicity")
MATCH_RESULT_KEYS = {
    "trial_id",
    "trial_name",
    "score",
    "matching_criteria",
    "failing_criteria",
}


@given(molecule=molecule_strategy, binding_site=binding_site_strategy)
@invariant_settings
def test_dock_returns_well_formed_docking_result(molecule, binding_site):
    result = MolecularDocking(receptor_pdb="receptor_1.pdb").dock(molecule, binding_site)
    assert isinstance(result, DockingResult)
    assert result.ligand is molecule
    assert result.receptor == "receptor_1.pdb"
    assert type(result.binding_energy) is float
    assert math.isfinite(result.binding_energy)
    assert isinstance(result.pose, list)
    assert len(result.pose) == 5
    for coordinate in result.pose:
        assert isinstance(coordinate, list)
        assert len(coordinate) == 3
        assert all(type(value) is float for value in coordinate)
    assert isinstance(result.interactions, list)
    for interaction in result.interactions:
        assert isinstance(interaction, dict)


@given(molecule=molecule_strategy, binding_site=binding_site_strategy)
@invariant_settings
def test_dock_binding_energy_is_a_finite_float(molecule, binding_site):
    result = MolecularDocking().dock(molecule, binding_site)
    assert type(result.binding_energy) is float
    assert math.isfinite(result.binding_energy)


@given(molecule=molecule_strategy)
@invariant_settings
def test_predict_returns_admet_prediction_with_expected_keys(molecule):
    result = ADMETPredictor().predict(molecule)
    assert isinstance(result, ADMETPrediction)
    assert result.molecule is molecule
    data = asdict(result)
    assert set(data) >= {"molecule", *ADMET_SCORE_FIELDS, *ADMET_FLAG_FIELDS}
    for field in ADMET_SCORE_FIELDS:
        assert isinstance(data[field], (int, float))
        assert math.isfinite(data[field])
        assert 0.0 <= data[field] <= 1.0
    for field in ADMET_FLAG_FIELDS:
        assert type(data[field]) is bool


@given(molecule=molecule_strategy)
@invariant_settings
def test_drug_likeness_score_is_a_float_in_unit_interval(molecule):
    score = ADMETPredictor().drug_likeness_score(molecule)
    assert type(score) is float
    assert 0.0 <= score <= 1.0


@given(patient=patient_strategy, trials=st.lists(trial_strategy, max_size=10))
@invariant_settings
def test_match_returns_list_of_scored_trials(patient, trials):
    matches = ClinicalTrialMatcher().match(patient, trials)
    assert isinstance(matches, list)
    for match in matches:
        assert isinstance(match, dict)
        assert set(match) >= MATCH_RESULT_KEYS
        assert type(match["score"]) is float
        assert 0.0 < match["score"] <= 1.0
        assert isinstance(match["matching_criteria"], list)
        assert isinstance(match["failing_criteria"], list)
    scores = [match["score"] for match in matches]
    assert scores == sorted(scores, reverse=True)


@given(patient=patient_strategy, trial=trial_strategy)
@invariant_settings
def test_calculate_match_score_returns_float(patient, trial):
    score = ClinicalTrialMatcher()._calculate_match_score(patient, trial)
    assert type(score) is float
    assert 0.0 <= score <= 1.0


@given(
    library=st.lists(molecule_strategy, max_size=15),
    binding_site=binding_site_strategy,
    top_n=st.integers(min_value=1, max_value=10),
)
@invariant_settings
def test_virtual_screen_returns_sorted_top_n_results(library, binding_site, top_n):
    results = MolecularDocking().virtual_screen(library, binding_site, top_n)
    assert isinstance(results, list)
    assert len(results) <= min(top_n, len(library))
    assert all(isinstance(r, DockingResult) for r in results)
    energies = [r.binding_energy for r in results]
    assert energies == sorted(energies)
