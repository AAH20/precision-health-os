"""Tests for integration module."""

from datetime import datetime

from precision_health_os.integration import EHRConnector, EventBus, FHIRConverter, HL7v2Parser


class TestFHIRConverter:
    """Tests for FHIRConverter."""

    def setup_method(self) -> None:
        self.converter = FHIRConverter()

    def test_patient_to_fhir(self) -> None:
        patient_data = {
            "id": "p001",
            "mrn": "MRN12345",
            "name": "John Doe",
            "sex": "male",
            "date_of_birth": datetime(1990, 1, 15),
        }
        result = self.converter.patient_to_fhir(patient_data)
        assert result.resource_type == "Patient"
        assert result.data["resourceType"] == "Patient"
        assert result.data["gender"] == "male"

    def test_observation_to_fhir(self) -> None:
        observation = {
            "patient_id": "p001",
            "name": "Heart Rate",
            "loinc_code": "8867-4",
            "value": 72,
            "unit": "bpm",
            "timestamp": datetime(2026, 1, 1, 12, 0, 0),
        }
        result = self.converter.observation_to_fhir(observation)
        assert result.resource_type == "Observation"
        assert result.data["valueQuantity"]["value"] == 72

    def test_medication_to_fhir(self) -> None:
        medication = {
            "patient_id": "p001",
            "name": "Metformin",
            "dosage": "500mg twice daily",
            "route": "oral",
        }
        result = self.converter.medication_to_fhir(medication)
        assert result.resource_type == "MedicationRequest"
        assert result.data["medicationCodeableConcept"]["text"] == "Metformin"

    def test_fhir_to_patient(self) -> None:
        fhir_data = {
            "id": "p001",
            "name": [{"given": ["John"], "family": "Doe"}],
            "gender": "male",
            "birthDate": "1990-01-15",
            "identifier": [{"system": "http://hospital.smarthealth.org/mrn", "value": "MRN12345"}],
        }
        result = self.converter.fhir_to_patient(fhir_data)
        assert result["id"] == "p001"
        assert result["name"] == "John Doe"
        assert result["mrn"] == "MRN12345"


class TestHL7v2Parser:
    """Tests for HL7v2Parser."""

    def setup_method(self) -> None:
        self.parser = HL7v2Parser()

    def test_parse_message(self) -> None:
        message = (
            "MSH|^~\\&|PHOS|HOSPITAL|EHR|HOSPITAL|20260101120000||ADT^A08|MSG001|P|2.5\r"
            "PID|1||MRN12345^p001||Doe^John||19900115|M"
        )
        result = self.parser.parse(message)
        assert result["message_type"] == "ADT^A08"
        assert result["patient_id"] == "MRN12345^p001"

    def test_create_adt(self) -> None:
        patient = {
            "id": "p001",
            "mrn": "MRN12345",
            "name": "Doe^John",
            "date_of_birth": datetime(1990, 1, 15),
            "sex": "M",
        }
        message = self.parser.create_adt(patient)
        assert "MSH|" in message
        "PID|" in message


class TestEventBus:
    """Tests for EventBus."""

    def setup_method(self) -> None:
        self.bus = EventBus()

    def test_subscribe_and_publish(self) -> None:
        received = []
        self.bus.subscribe("test.event", lambda e: received.append(e))
        self.bus.publish("test.event", {"key": "value"})
        assert len(received) == 1
        assert received[0]["payload"]["key"] == "value"

    def test_unsubscribe(self) -> None:
        received = []
        handler = lambda e: received.append(e)
        self.bus.subscribe("test.event", handler)
        self.bus.unsubscribe("test.event", handler)
        self.bus.publish("test.event", {})
        assert len(received) == 0

    def test_get_events(self) -> None:
        self.bus.publish("type1", {"a": 1})
        self.bus.publish("type2", {"b": 2})
        self.bus.publish("type1", {"a": 3})
        events = self.bus.get_events(event_type="type1")
        assert len(events) == 2


class TestEHRConnector:
    """Tests for EHRConnector."""

    def setup_method(self) -> None:
        self.converter = FHIRConverter()
        self.connector = EHRConnector("https://fhir.example.com", self.converter)

    def test_connect(self) -> None:
        assert self.connector.connect() is True

    def test_push_observation(self) -> None:
        observation = {
            "patient_id": "p001",
            "name": "Heart Rate",
            "loinc_code": "8867-4",
            "value": 72,
            "unit": "bpm",
            "timestamp": datetime(2026, 1, 1),
        }
        assert self.connector.push_observation(observation) is True
