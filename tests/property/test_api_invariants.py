"""Property-based invariant tests for the Precision Health API."""

from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.models import AlertSeverity, Patient, VitalSigns

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

valid_sex = st.sampled_from(["male", "female", "other", "unknown"])
valid_severity = st.sampled_from(list(AlertSeverity))

patient_strategy = st.builds(
    Patient,
    id=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    mrn=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    name=st.text(min_size=1, max_size=50),
    date_of_birth=st.datetimes(
        min_value=datetime(1900, 1, 1, tzinfo=UTC),
        max_value=datetime(2025, 12, 31, tzinfo=UTC),
    ),
    sex=valid_sex,
)

vitals_strategy = st.builds(
    VitalSigns,
    patient_id=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    heart_rate_bpm=st.one_of(st.none(), st.floats(min_value=0, max_value=300, allow_nan=False)),
    blood_pressure_systolic=st.one_of(
        st.none(), st.floats(min_value=0, max_value=300, allow_nan=False)
    ),
    blood_pressure_diastolic=st.one_of(
        st.none(), st.floats(min_value=0, max_value=200, allow_nan=False)
    ),
    spo2_percent=st.one_of(st.none(), st.floats(min_value=0, max_value=100, allow_nan=False)),
    temperature_celsius=st.one_of(
        st.none(), st.floats(min_value=30, max_value=45, allow_nan=False)
    ),
    respiratory_rate=st.one_of(st.none(), st.floats(min_value=0, max_value=100, allow_nan=False)),
    glucose_mg_dl=st.one_of(st.none(), st.floats(min_value=0, max_value=1000, allow_nan=False)),
)

text_id = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
)


# ---------------------------------------------------------------------------
# Invariant: register_patient returns a Patient with the same id
# ---------------------------------------------------------------------------


@given(patient=patient_strategy)
@settings(max_examples=30, deadline=None)
def test_register_patient_returns_same_id(patient: Patient) -> None:
    api = PrecisionHealthAPI()
    result = api.register_patient(patient)
    assert isinstance(result, Patient)
    assert result.id == patient.id


# ---------------------------------------------------------------------------
# Invariant: ingest_vitals returns a list (possibly empty)
# ---------------------------------------------------------------------------


@given(vitals=vitals_strategy)
@settings(max_examples=30, deadline=None)
def test_ingest_vitals_returns_list(vitals: VitalSigns) -> None:
    api = PrecisionHealthAPI()
    result = api.ingest_vitals(vitals)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Invariant: get_active_alerts returns only unacknowledged alerts
# ---------------------------------------------------------------------------


@given(
    patient=patient_strategy,
    vitals_list=st.lists(vitals_strategy, min_size=0, max_size=5),
)
@settings(max_examples=30, deadline=None)
def test_get_active_alerts_only_unacknowledged(
    patient: Patient, vitals_list: list[VitalSigns]
) -> None:
    api = PrecisionHealthAPI()
    api.register_patient(patient)
    for v in vitals_list:
        api.ingest_vitals(v)

    active = api.get_active_alerts()
    assert isinstance(active, list)
    for alert in active:
        assert alert.acknowledged is False


# ---------------------------------------------------------------------------
# Invariant: acknowledge_alert returns bool
# ---------------------------------------------------------------------------


@given(
    patient=patient_strategy,
    vitals_list=st.lists(vitals_strategy, min_size=0, max_size=5),
    user_id=text_id,
)
@settings(max_examples=30, deadline=None)
def test_acknowledge_alert_returns_bool(
    patient: Patient, vitals_list: list[VitalSigns], user_id: str
) -> None:
    api = PrecisionHealthAPI()
    api.register_patient(patient)
    for v in vitals_list:
        api.ingest_vitals(v)

    active = api.get_active_alerts()
    alert_id = active[0].id if active else "nonexistent-alert-id"

    result = api.acknowledge_alert(alert_id, user_id)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Invariant: check_permission returns bool
# ---------------------------------------------------------------------------


@given(user_id=text_id, permission=text_id)
@settings(max_examples=30, deadline=None)
def test_check_permission_returns_bool(user_id: str, permission: str) -> None:
    api = PrecisionHealthAPI()
    result = api.check_permission(user_id, permission)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Invariant: get_system_status returns a dict with expected keys
# ---------------------------------------------------------------------------


@given(
    patient=patient_strategy,
    vitals_list=st.lists(vitals_strategy, min_size=0, max_size=3),
)
@settings(max_examples=30, deadline=None)
def test_get_system_status_returns_dict_with_keys(
    patient: Patient, vitals_list: list[VitalSigns]
) -> None:
    api = PrecisionHealthAPI()
    api.register_patient(patient)
    for v in vitals_list:
        api.ingest_vitals(v)

    status = api.get_system_status()
    assert isinstance(status, dict)
    expected_keys = {
        "patients_registered",
        "active_alerts",
        "audit_events",
        "event_bus_events",
    }
    assert expected_keys.issubset(status.keys())
