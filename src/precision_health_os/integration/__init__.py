"""Integration module: FHIR/HL7 interoperability, EHR integration, event bus."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from precision_health_os.models import FHIRResource
from precision_health_os.utils import generate_id, utcnow

logger = logging.getLogger(__name__)


class FHIRConverter:
    """Convert internal models to/from FHIR R4 resources."""

    def patient_to_fhir(self, patient_data: dict[str, Any]) -> FHIRResource:
        """Convert patient dict to FHIR Patient resource."""
        fhir_patient = {
            "resourceType": "Patient",
            "id": patient_data.get("id"),
            "identifier": [
                {
                    "system": "http://hospital.smarthealth.org/mrn",
                    "value": patient_data.get("mrn", ""),
                }
            ],
            "name": [{"use": "official", "text": patient_data.get("name", "")}],
            "gender": patient_data.get("sex", "unknown"),
            "birthDate": patient_data.get("date_of_birth", "").strftime("%Y-%m-%d")
            if isinstance(patient_data.get("date_of_birth"), datetime)
            else patient_data.get("date_of_birth", ""),
        }
        return FHIRResource(resource_type="Patient", id=patient_data.get("id"), data=fhir_patient)

    def observation_to_fhir(self, observation: dict[str, Any]) -> FHIRResource:
        """Convert observation to FHIR Observation resource."""
        fhir_obs = {
            "resourceType": "Observation",
            "id": observation.get("id", generate_id()),
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": observation.get("loinc_code", ""),
                        "display": observation.get("name", ""),
                    }
                ]
            },
            "subject": {"reference": f"Patient/{observation.get('patient_id', '')}"},
            "effectiveDateTime": observation.get("timestamp", utcnow()).isoformat()
            if isinstance(observation.get("timestamp"), datetime)
            else observation.get("timestamp", ""),
            "valueQuantity": {
                "value": observation.get("value"),
                "unit": observation.get("unit", ""),
                "system": "http://unitsofmeasure.org",
            },
        }
        return FHIRResource(
            resource_type="Observation",
            id=fhir_obs["id"],
            data=fhir_obs,
        )

    def medication_to_fhir(self, medication: dict[str, Any]) -> FHIRResource:
        """Convert medication to FHIR MedicationRequest resource."""
        fhir_med = {
            "resourceType": "MedicationRequest",
            "id": medication.get("id", generate_id()),
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {"text": medication.get("name", "")},
            "subject": {"reference": f"Patient/{medication.get('patient_id', '')}"},
            "dosageInstruction": [
                {
                    "text": medication.get("dosage", ""),
                    "route": {"text": medication.get("route", "oral")},
                }
            ],
        }
        return FHIRResource(
            resource_type="MedicationRequest",
            id=fhir_med["id"],
            data=fhir_med,
        )

    def fhir_to_patient(self, fhir_data: dict[str, Any]) -> dict[str, Any]:
        """Convert FHIR Patient resource to internal patient dict."""
        name = ""
        if fhir_data.get("name"):
            name_parts = [
                *fhir_data["name"][0].get("given", []),
                fhir_data["name"][0].get("family", ""),
            ]
            name = " ".join(filter(None, name_parts))

        mrn = ""
        for ident in fhir_data.get("identifier", []):
            if "mrn" in ident.get("system", ""):
                mrn = ident.get("value", "")
                break

        return {
            "id": fhir_data.get("id", ""),
            "mrn": mrn,
            "name": name,
            "sex": fhir_data.get("gender", "unknown"),
            "date_of_birth": fhir_data.get("birthDate", ""),
        }


class HL7v2Parser:
    """Parse HL7 v2.x messages."""

    def parse(self, message: str) -> dict[str, Any]:
        """Parse an HL7 v2 message."""
        segments = message.strip().split("\r")
        result: dict[str, Any] = {"segments": []}

        for segment in segments:
            if not segment:
                continue
            fields = segment.split("|")
            segment_type = fields[0]
            result["segments"].append({"type": segment_type, "fields": fields[1:]})

            if segment_type == "MSH":
                result["message_type"] = fields[8] if len(fields) > 8 else ""
                result["sending_app"] = fields[2] if len(fields) > 2 else ""
            elif segment_type == "PID":
                result["patient_id"] = fields[3] if len(fields) > 3 else ""
                # Unescape \r and \n that were escaped during creation
                name = fields[5] if len(fields) > 5 else ""
                name = name.replace("\\r", "\r").replace("\\n", "\n")
                result["patient_name"] = name

        return result

    def create_adt(self, patient: dict[str, Any], event_type: str = "A08") -> str:
        """Create an HL7 ADT message."""
        timestamp = utcnow().strftime("%Y%m%d%H%M%S")
        dob = patient.get("date_of_birth")
        dob_str = dob.strftime("%Y%m%d") if isinstance(dob, datetime) else ""
        # Escape \r and \n in names so they don't break HL7 segment delimiters
        name = patient.get("name", "").replace("\r", "\\r").replace("\n", "\\n")
        pid = (
            f"PID|1||{patient.get('mrn', '')}^{patient.get('id', '')}"
            f"||{name}||{dob_str}|{patient.get('sex', 'U')}"
        )
        segments = [
            f"MSH|^~\\&|PHOS|HOSPITAL|EHR|HOSPITAL|{timestamp}||ADT^{event_type}|{generate_id()}|P|2.5",
            pid,
        ]
        return "\r".join(segments)


class EventBus:
    """Async event bus for internal service communication."""

    def __init__(self) -> None:
        """Initialize the event bus."""
        self._subscribers: dict[str, list[Any]] = {}
        self._event_log: list[dict[str, Any]] = []

    def subscribe(self, event_type: str, handler: Any) -> None:
        """Subscribe a handler to an event type."""
        self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: Any) -> None:
        """Unsubscribe a handler."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                h for h in self._subscribers[event_type] if h != handler
            ]

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        """Publish an event."""
        event = {
            "id": generate_id(),
            "type": event_type,
            "payload": payload,
            "timestamp": utcnow().isoformat(),
        }
        self._event_log.append(event)

        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler failed for {event_type}: {e}")

    def get_events(self, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent events."""
        events = self._event_log
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        return events[-limit:]


class EHRConnector:
    """EHR system connector with FHIR/HL7 support."""

    def __init__(self, base_url: str, fhir_converter: FHIRConverter) -> None:
        """Initialize the EHR connector."""
        self.base_url = base_url
        self.fhir = fhir_converter
        self._connected = False

    def connect(self) -> bool:
        """Test connection to EHR system."""
        # Production: implement OAuth2 + SMART on FHIR
        self._connected = True
        return True

    def fetch_patient(self, patient_id: str) -> dict[str, Any] | None:
        """Fetch patient from EHR via FHIR."""
        # Production: HTTP GET to FHIR endpoint
        return None

    def push_observation(self, observation: dict[str, Any]) -> bool:
        """Push observation to EHR via FHIR."""
        self.fhir.observation_to_fhir(observation)
        # Production: HTTP POST to FHIR endpoint
        return True

    def push_medication(self, medication: dict[str, Any]) -> bool:
        """Push medication to EHR via FHIR."""
        self.fhir.medication_to_fhir(medication)
        # Production: HTTP POST to FHIR endpoint
        return True
