r"""Regression tests for bugs found by Wave 1 property tests.

Bug 1: ADMETPredictor.predict returns int 0 instead of float when
        Lipinski violations >= 4 (max(0, ...) returns int).
Bug 2: HL7v2Parser round-trip breaks when patient name contains \\r
        (parsed as empty string).
"""

from __future__ import annotations

from precision_health_os.drug_discovery import ADMETPredictor, Molecule
from precision_health_os.integration import HL7v2Parser


class TestADMETReturnType:
    """Bug 1: max(0, ...) returns int 0, not float."""

    def test_bioavailability_is_float_when_zero(self) -> None:
        """When violations >= 4, bioavailability should be 0.0 (float), not 0 (int)."""
        mol = Molecule(
            smiles="0",
            name="",
            molecular_weight=501.0,
            logp=6.0,
            hbd=6,
            hba=11,
            tpsa=0.0,
            rotatable_bonds=0,
        )
        predictor = ADMETPredictor()
        result = predictor.predict(mol)
        assert isinstance(result.bioavailability, float), (
            f"bioavailability is {type(result.bioavailability).__name__}, expected float"
        )
        assert result.bioavailability == 0.0

    def test_absorption_is_float_when_zero(self) -> None:
        """When violations >= 5, absorption should be 0.0 (float), not 0 (int)."""
        mol = Molecule(
            smiles="0",
            name="",
            molecular_weight=501.0,
            logp=6.0,
            hbd=6,
            hba=11,
            tpsa=0.0,
            rotatable_bonds=0,
        )
        predictor = ADMETPredictor()
        result = predictor.predict(mol)
        assert isinstance(result.absorption, float), (
            f"absorption is {type(result.absorption).__name__}, expected float"
        )


class TestHL7RoundTrip:
    r"""Bug 2: \\r in patient name breaks HL7v2 round-trip."""

    def test_name_with_carriage_return_round_trips(self) -> None:
        parser = HL7v2Parser()
        patient = {"id": "P001", "name": "John\rDoe", "dob": "19900101", "sex": "M"}
        msg = parser.create_adt(patient)
        parsed = parser.parse(msg)
        assert parsed.get("patient_name") == "John\rDoe", (
            f"expected 'John\\rDoe', got '{parsed.get('patient_name')}'"
        )

    def test_name_with_newline_round_trips(self) -> None:
        parser = HL7v2Parser()
        patient = {"id": "P001", "name": "John\nDoe", "dob": "19900101", "sex": "M"}
        msg = parser.create_adt(patient)
        parsed = parser.parse(msg)
        assert parsed.get("patient_name") == "John\nDoe", (
            f"expected 'John\\nDoe', got '{parsed.get('patient_name')}'"
        )
