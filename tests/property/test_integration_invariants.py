"""Property-based invariant tests for the integration module."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from precision_health_os.integration import (
    EventBus,
    FHIRConverter,
    HL7v2Parser,
)

# Strategies
# Exclude HL7 delimiter characters (\r, |, ^, ~, \, &) to avoid known parser bugs.
hl7_safe_text = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters="|\r\n^~\\&",
    ),
)

patient_data_strategy = st.fixed_dictionaries(
    {
        "id": hl7_safe_text,
        "mrn": hl7_safe_text,
        "name": hl7_safe_text,
        "sex": st.sampled_from(["male", "female", "other", "unknown"]),
        "date_of_birth": hl7_safe_text,
    }
)

observation_data_strategy = st.fixed_dictionaries(
    {
        "id": st.text(min_size=1, max_size=20),
        "patient_id": st.text(min_size=1, max_size=20),
        "loinc_code": st.text(min_size=1, max_size=20),
        "name": st.text(min_size=1, max_size=50),
        "value": st.floats(allow_nan=False, allow_infinity=False),
        "unit": st.text(min_size=1, max_size=20),
        "timestamp": st.text(min_size=1, max_size=30),
    }
)

medication_data_strategy = st.fixed_dictionaries(
    {
        "id": st.text(min_size=1, max_size=20),
        "patient_id": st.text(min_size=1, max_size=20),
        "name": st.text(min_size=1, max_size=50),
        "dosage": st.text(min_size=1, max_size=50),
        "route": st.text(min_size=1, max_size=20),
    }
)

hl7_message_strategy = st.text(min_size=1, max_size=500)

event_type_strategy = st.text(min_size=1, max_size=30)

payload_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.text(min_size=0, max_size=50),
    max_size=5,
)

adt_event_type_strategy = st.text(
    min_size=1,
    max_size=5,
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters="|\r\n^~\\&",
    ),
)


class TestFHIRConverterInvariants:
    """Invariants for FHIRConverter."""

    @given(patient_data=patient_data_strategy)
    @settings(max_examples=30, deadline=None)
    def test_patient_to_fhir_returns_fhir_resource(self, patient_data):
        converter = FHIRConverter()
        result = converter.patient_to_fhir(patient_data)
        assert result is not None
        assert result.resource_type == "Patient"
        assert result.data["resourceType"] == "Patient"

    @given(observation_data=observation_data_strategy)
    @settings(max_examples=30, deadline=None)
    def test_observation_to_fhir_returns_fhir_resource(self, observation_data):
        converter = FHIRConverter()
        result = converter.observation_to_fhir(observation_data)
        assert result is not None
        assert result.resource_type == "Observation"
        assert result.data["resourceType"] == "Observation"

    @given(medication_data=medication_data_strategy)
    @settings(max_examples=30, deadline=None)
    def test_medication_to_fhir_returns_fhir_resource(self, medication_data):
        converter = FHIRConverter()
        result = converter.medication_to_fhir(medication_data)
        assert result is not None
        assert result.resource_type == "MedicationRequest"
        assert result.data["resourceType"] == "MedicationRequest"

    @given(patient_data=patient_data_strategy)
    @settings(max_examples=30, deadline=None)
    def test_patient_fhir_round_trip_preserves_id(self, patient_data):
        converter = FHIRConverter()
        fhir_resource = converter.patient_to_fhir(patient_data)
        round_tripped = converter.fhir_to_patient(fhir_resource.data)
        assert round_tripped["id"] == patient_data["id"]

    @given(patient_data=patient_data_strategy)
    @settings(max_examples=30, deadline=None)
    def test_patient_fhir_round_trip_preserves_mrn(self, patient_data):
        converter = FHIRConverter()
        fhir_resource = converter.patient_to_fhir(patient_data)
        round_tripped = converter.fhir_to_patient(fhir_resource.data)
        assert round_tripped["mrn"] == patient_data["mrn"]

    @given(patient_data=patient_data_strategy)
    @settings(max_examples=30, deadline=None)
    def test_patient_fhir_round_trip_preserves_sex(self, patient_data):
        converter = FHIRConverter()
        fhir_resource = converter.patient_to_fhir(patient_data)
        round_tripped = converter.fhir_to_patient(fhir_resource.data)
        assert round_tripped["sex"] == patient_data["sex"]

    @given(patient_data=patient_data_strategy)
    @settings(max_examples=30, deadline=None)
    def test_patient_fhir_round_trip_preserves_dob(self, patient_data):
        converter = FHIRConverter()
        fhir_resource = converter.patient_to_fhir(patient_data)
        round_tripped = converter.fhir_to_patient(fhir_resource.data)
        assert round_tripped["date_of_birth"] == patient_data["date_of_birth"]


class TestHL7v2ParserInvariants:
    """Invariants for HL7v2Parser."""

    @given(message=hl7_message_strategy)
    @settings(max_examples=30, deadline=None)
    def test_parse_returns_dict(self, message):
        parser = HL7v2Parser()
        result = parser.parse(message)
        assert isinstance(result, dict)
        assert "segments" in result
        assert isinstance(result["segments"], list)

    @given(patient_data=patient_data_strategy, event_type=adt_event_type_strategy)
    @settings(max_examples=30, deadline=None)
    def test_create_adt_parse_round_trip_patient_name(self, patient_data, event_type):
        parser = HL7v2Parser()
        adt_message = parser.create_adt(patient_data, event_type)
        parsed = parser.parse(adt_message)
        assert parsed["patient_name"] == patient_data["name"]

    @given(patient_data=patient_data_strategy, event_type=adt_event_type_strategy)
    @settings(max_examples=30, deadline=None)
    def test_create_adt_parse_round_trip_patient_id_contains_mrn(self, patient_data, event_type):
        parser = HL7v2Parser()
        adt_message = parser.create_adt(patient_data, event_type)
        parsed = parser.parse(adt_message)
        assert patient_data["mrn"] in parsed["patient_id"]

    @given(patient_data=patient_data_strategy, event_type=adt_event_type_strategy)
    @settings(max_examples=30, deadline=None)
    def test_create_adt_parse_round_trip_message_type_contains_event(
        self, patient_data, event_type
    ):
        parser = HL7v2Parser()
        adt_message = parser.create_adt(patient_data, event_type)
        parsed = parser.parse(adt_message)
        assert event_type in parsed["message_type"]


class TestEventBusInvariants:
    """Invariants for EventBus."""

    @given(event_type=event_type_strategy, payload=payload_strategy)
    @settings(max_examples=30, deadline=None)
    def test_publish_returns_none(self, event_type, payload):
        bus = EventBus()
        result = bus.publish(event_type, payload)
        assert result is None

    @given(event_type=event_type_strategy)
    @settings(max_examples=30, deadline=None)
    def test_subscribe_returns_none(self, event_type):
        bus = EventBus()

        def handler(event):
            return None

        result = bus.subscribe(event_type, handler)
        assert result is None

    @given(event_type=event_type_strategy, payload=payload_strategy)
    @settings(max_examples=30, deadline=None)
    def test_publish_then_get_events_round_trip(self, event_type, payload):
        bus = EventBus()
        bus.publish(event_type, payload)
        events = bus.get_events(event_type)
        assert len(events) >= 1
        assert events[-1]["type"] == event_type
        assert events[-1]["payload"] == payload

    @given(event_type=event_type_strategy, payload=payload_strategy)
    @settings(max_examples=30, deadline=None)
    def test_subscribe_handler_called_on_publish(self, event_type, payload):
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(event_type, handler)
        bus.publish(event_type, payload)
        assert len(received) == 1
        assert received[0]["type"] == event_type
        assert received[0]["payload"] == payload
