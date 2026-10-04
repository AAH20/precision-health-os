"""Tests for integration module gaps."""

from __future__ import annotations

from datetime import UTC, datetime

from precision_health_os.integration import (
    EHRConnector,
    EventBus,
    FHIRConverter,
    HL7v2Parser,
)


class TestFHIRConverter:
    """Tests for FHIRConverter."""

    def setup_method(self) -> None:
        self.converter = FHIRConverter()

    def test_patient_to_fhir(self) -> None:
        patient = {
            "id": "p1",
            "mrn": "MRN123",
            "name": "John Doe",
            "sex": "male",
            "date_of_birth": datetime(1990, 1, 1, tzinfo=UTC),
        }
        result = self.converter.patient_to_fhir(patient)
        assert result.resource_type == "Patient"
        assert result.id == "p1"
        assert result.data["resourceType"] == "Patient"
        assert result.data["name"][0]["text"] == "John Doe"
        assert result.data["gender"] == "male"
        assert result.data["birthDate"] == "1990-01-01"

    def test_observation_to_fhir(self) -> None:
        obs = {
            "id": "o1",
            "patient_id": "p1",
            "loinc_code": "8867-4",
            "name": "Heart Rate",
            "value": 72,
            "unit": "bpm",
            "timestamp": datetime(2024, 1, 1, tzinfo=UTC),
        }
        result = self.converter.observation_to_fhir(obs)
        assert result.resource_type == "Observation"
        assert result.id == "o1"
        assert result.data["status"] == "final"
        assert result.data["code"]["coding"][0]["code"] == "8867-4"
        assert result.data["subject"]["reference"] == "Patient/p1"
        assert result.data["valueQuantity"]["value"] == 72

    def test_medication_to_fhir(self) -> None:
        med = {
            "id": "m1",
            "patient_id": "p1",
            "name": "Aspirin",
            "dosage": "81mg daily",
            "route": "oral",
        }
        result = self.converter.medication_to_fhir(med)
        assert result.resource_type == "MedicationRequest"
        assert result.id == "m1"
        assert result.data["status"] == "active"
        assert result.data["medicationCodeableConcept"]["text"] == "Aspirin"
        assert result.data["dosageInstruction"][0]["text"] == "81mg daily"

    def test_fhir_to_patient_full(self) -> None:
        fhir = {
            "id": "p1",
            "name": [{"given": ["John"], "family": "Doe"}],
            "identifier": [{"system": "http://hospital.smarthealth.org/mrn", "value": "MRN123"}],
            "gender": "male",
            "birthDate": "1990-01-01",
        }
        result = self.converter.fhir_to_patient(fhir)
        assert result["id"] == "p1"
        assert result["mrn"] == "MRN123"
        assert result["name"] == "John Doe"
        assert result["sex"] == "male"
        assert result["date_of_birth"] == "1990-01-01"

    def test_fhir_to_patient_no_name(self) -> None:
        fhir = {
            "id": "p2",
            "identifier": [{"system": "http://hospital.smarthealth.org/mrn", "value": "MRN456"}],
            "gender": "female",
            "birthDate": "1985-05-05",
        }
        result = self.converter.fhir_to_patient(fhir)
        assert result["name"] == ""
        assert result["mrn"] == "MRN456"

    def test_fhir_to_patient_no_mrn(self) -> None:
        fhir = {
            "id": "p3",
            "name": [{"given": ["Jane"], "family": "Smith"}],
            "identifier": [{"system": "http://example.org", "value": "X123"}],
            "gender": "female",
            "birthDate": "1980-03-15",
        }
        result = self.converter.fhir_to_patient(fhir)
        assert result["mrn"] == ""
        assert result["name"] == "Jane Smith"

    def test_fhir_to_patient_name_given_only(self) -> None:
        fhir = {
            "id": "p4",
            "name": [{"given": ["Alice"]}],
            "gender": "unknown",
        }
        result = self.converter.fhir_to_patient(fhir)
        assert result["name"] == "Alice"

    def test_fhir_to_patient_name_family_only(self) -> None:
        fhir = {
            "id": "p5",
            "name": [{"family": "Johnson"}],
            "gender": "unknown",
        }
        result = self.converter.fhir_to_patient(fhir)
        assert result["name"] == "Johnson"


class TestHL7v2Parser:
    """Tests for HL7v2Parser."""

    def setup_method(self) -> None:
        self.parser = HL7v2Parser()

    def test_parse_multi_segment(self) -> None:
        msg = (
            "MSH|^~\\&|PHOS|HOSPITAL|EHR|HOSPITAL|20240101120000||ADT^A08|12345|P|2.5\r"
            "PID|1||MRN123^P1||Doe^John||19900101|M\r"
            "OBX|1|NM|8867-4^Heart Rate||72|bpm"
        )
        result = self.parser.parse(msg)
        assert result["message_type"] == "ADT^A08"
        assert result["sending_app"] == "PHOS"
        assert result["patient_id"] == "MRN123^P1"
        assert result["patient_name"] == "Doe^John"
        assert len(result["segments"]) == 3

    def test_parse_empty_message(self) -> None:
        result = self.parser.parse("")
        assert result == {"segments": []}

    def test_parse_missing_pid(self) -> None:
        msg = "MSH|^~\\&|PHOS|HOSPITAL|EHR|HOSPITAL|20240101120000||ADT^A08|12345|P|2.5"
        result = self.parser.parse(msg)
        assert result["message_type"] == "ADT^A08"
        assert "patient_id" not in result
        assert "patient_name" not in result

    def test_create_adt(self) -> None:
        patient = {
            "id": "p1",
            "mrn": "MRN123",
            "name": "John Doe",
            "date_of_birth": datetime(1990, 1, 1, tzinfo=UTC),
            "sex": "male",
        }
        msg = self.parser.create_adt(patient)
        assert "MSH|" in msg
        assert "PID|1||MRN123^p1||John Doe||19900101|male" in msg


class TestEventBus:
    """Tests for EventBus."""

    def setup_method(self) -> None:
        self.bus = EventBus()

    def test_subscribe_and_publish(self) -> None:
        received = []
        self.bus.subscribe("test.event", lambda e: received.append(e))
        self.bus.publish("test.event", {"key": "value"})
        assert len(received) == 1
        assert received[0]["type"] == "test.event"
        assert received[0]["payload"] == {"key": "value"}

    def test_unsubscribe(self) -> None:
        received = []

        def handler(e: dict) -> None:
            received.append(e)

        self.bus.subscribe("test.event", handler)
        self.bus.unsubscribe("test.event", handler)
        self.bus.publish("test.event", {})
        assert len(received) == 0

    def test_unsubscribe_nonexistent_event(self) -> None:
        def handler(e: dict) -> None:
            pass

        self.bus.unsubscribe("nonexistent", handler)

    def test_publish_handler_exception(self) -> None:
        def bad_handler(e: dict) -> None:
            raise RuntimeError("boom")

        self.bus.subscribe("test.event", bad_handler)
        self.bus.publish("test.event", {})

    def test_get_events_no_filter(self) -> None:
        self.bus.publish("event.a", {"n": 1})
        self.bus.publish("event.b", {"n": 2})
        events = self.bus.get_events()
        assert len(events) == 2

    def test_get_events_with_filter(self) -> None:
        self.bus.publish("event.a", {"n": 1})
        self.bus.publish("event.b", {"n": 2})
        self.bus.publish("event.a", {"n": 3})
        events = self.bus.get_events("event.a")
        assert len(events) == 2
        assert all(e["type"] == "event.a" for e in events)

    def test_get_events_limit(self) -> None:
        for i in range(5):
            self.bus.publish("event.a", {"n": i})
        events = self.bus.get_events(limit=3)
        assert len(events) == 3


class TestEHRConnector:
    """Tests for EHRConnector."""

    def setup_method(self) -> None:
        self.converter = FHIRConverter()
        self.connector = EHRConnector("https://example.com/fhir", self.converter)

    def test_connect(self) -> None:
        assert self.connector.connect() is True

    def test_fetch_patient(self) -> None:
        assert self.connector.fetch_patient("p1") is None

    def test_push_observation(self) -> None:
        obs = {
            "patient_id": "p1",
            "loinc_code": "8867-4",
            "name": "Heart Rate",
            "value": 72,
            "unit": "bpm",
        }
        assert self.connector.push_observation(obs) is True

    def test_push_medication(self) -> None:
        med = {
            "patient_id": "p1",
            "name": "Aspirin",
            "dosage": "81mg daily",
        }
        assert self.connector.push_medication(med) is True
