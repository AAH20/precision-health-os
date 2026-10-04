"""Property-based invariant tests for the clinical module."""

from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from precision_health_os.clinical import AlertManager, CDSSEngine
from precision_health_os.models import (
    AlertSeverity,
    ClinicalAlert,
    Patient,
    VitalSigns,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

patient_id_st = st.text(min_size=1, max_size=20)
source_st = st.text(min_size=1, max_size=20)
title_st = st.text(min_size=1, max_size=50)
desc_st = st.text(min_size=1, max_size=100)
severity_st = st.sampled_from(list(AlertSeverity))


def vital_signs_st():
    return st.builds(
        VitalSigns,
        patient_id=patient_id_st,
        heart_rate_bpm=st.one_of(st.none(), st.floats(min_value=0, max_value=300)),
        blood_pressure_systolic=st.one_of(st.none(), st.floats(min_value=0, max_value=300)),
        blood_pressure_diastolic=st.one_of(st.none(), st.floats(min_value=0, max_value=200)),
        spo2_percent=st.one_of(st.none(), st.floats(min_value=0, max_value=100)),
        temperature_celsius=st.one_of(st.none(), st.floats(min_value=30, max_value=45)),
        respiratory_rate=st.one_of(st.none(), st.floats(min_value=0, max_value=100)),
        glucose_mg_dl=st.one_of(st.none(), st.floats(min_value=0, max_value=1000)),
    )


def alert_st():
    return st.builds(
        ClinicalAlert,
        id=st.text(min_size=1, max_size=20),
        patient_id=patient_id_st,
        severity=severity_st,
        title=title_st,
        description=desc_st,
        source=source_st,
        confidence=st.floats(min_value=0.0, max_value=1.0),
        recommendations=st.lists(st.text(min_size=1, max_size=50), max_size=3),
        acknowledged=st.booleans(),
    )


def patient_st():
    return st.builds(
        Patient,
        id=st.text(min_size=1, max_size=20),
        mrn=st.text(min_size=1, max_size=20),
        name=st.text(min_size=1, max_size=50),
        date_of_birth=st.datetimes(
            min_value=datetime(1900, 1, 1, tzinfo=UTC),
            max_value=datetime(2024, 12, 31, tzinfo=UTC),
        ),
        sex=st.sampled_from(["male", "female", "other", "unknown"]),
        weight_kg=st.one_of(st.none(), st.floats(min_value=0.1, max_value=500)),
        height_cm=st.one_of(st.none(), st.floats(min_value=0.1, max_value=300)),
        allergies=st.lists(st.text(min_size=1, max_size=20), max_size=3),
        medications=st.lists(st.text(min_size=1, max_size=20), max_size=3),
        conditions=st.lists(st.text(min_size=1, max_size=20), max_size=3),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluateVitals:
    @given(patient_id=patient_id_st, vitals=vital_signs_st())
    @settings(max_examples=30, deadline=None)
    def test_returns_list(self, patient_id, vitals):
        engine = CDSSEngine()
        result = engine.evaluate_vitals(patient_id, vitals)
        assert isinstance(result, list)


class TestEvaluateRules:
    @given(
        patient=patient_st(),
        context=st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_returns_list(self, patient, context):
        engine = CDSSEngine()
        result = engine.evaluate_rules(patient, context)
        assert isinstance(result, list)


class TestAlertManager:
    @given(
        alert=st.builds(
            ClinicalAlert,
            id=st.text(min_size=1, max_size=20),
            patient_id=patient_id_st,
            severity=severity_st,
            title=title_st,
            description=desc_st,
            source=source_st,
            confidence=st.floats(min_value=0.0, max_value=1.0),
            recommendations=st.lists(st.text(min_size=1, max_size=50), max_size=3),
            acknowledged=st.just(False),
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_add_alert_deduplicates_same_patient_source_title(self, alert):
        manager = AlertManager()
        first = manager.add_alert(alert)
        second_alert = alert.model_copy(update={"id": "different-id"})
        second = manager.add_alert(second_alert)
        assert first.id == second.id

    @given(alert=alert_st())
    @settings(max_examples=30, deadline=None)
    def test_add_alert_no_dedup_different_title(self, alert):
        manager = AlertManager()
        first = manager.add_alert(alert)
        second_alert = alert.model_copy(update={"id": "different-id", "title": "different-title"})
        second = manager.add_alert(second_alert)
        assert first.id != second.id

    @given(alert=alert_st())
    @settings(max_examples=30, deadline=None)
    def test_acknowledge_sets_acknowledged(self, alert):
        manager = AlertManager()
        added = manager.add_alert(alert)
        result = manager.acknowledge(added.id, "user1")
        assert result is not None
        assert result.acknowledged is True

    @given(alert=alert_st())
    @settings(max_examples=30, deadline=None)
    def test_get_active_alerts_returns_only_unacknowledged(self, alert):
        manager = AlertManager()
        added = manager.add_alert(alert)
        manager.acknowledge(added.id, "user1")
        active = manager.get_active_alerts()
        assert all(not a.acknowledged for a in active)
