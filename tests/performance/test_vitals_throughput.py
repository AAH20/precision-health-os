"""Performance tests for high-frequency vitals ingestion."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.iot import (
    AnomalyDetector,
    RemotePatientMonitor,
    SensorReading,
    WearableDataPipeline,
)
from precision_health_os.models import AlertSeverity, Patient, VitalSigns


def _make_patient(patient_id: str = "perf-patient-1") -> Patient:
    return Patient(
        id=patient_id,
        mrn="MRN-PERF-001",
        name="Performance Test Patient",
        date_of_birth=datetime(1980, 1, 1, tzinfo=UTC),
        sex="male",
    )


def _make_vitals(patient_id: str, hr: float = 72.0) -> VitalSigns:
    return VitalSigns(
        patient_id=patient_id,
        heart_rate_bpm=hr,
        spo2_percent=98.0,
        blood_pressure_systolic=120.0,
        blood_pressure_diastolic=80.0,
        temperature_celsius=37.0,
    )


def _make_reading(patient_id: str, value: float, ts: float) -> SensorReading:
    return SensorReading(
        device_id="device-001",
        patient_id=patient_id,
        sensor_type="heart_rate",
        value=value,
        timestamp=ts,
        unit="bpm",
    )


class TestHighFrequencyIngestion:
    """Performance tests for high-frequency vitals ingestion."""

    def test_100_readings_all_succeed(self) -> None:
        """Ingest 100 vitals readings for a single patient — all succeed."""
        api = PrecisionHealthAPI()
        patient = _make_patient()
        api.register_patient(patient)

        for i in range(100):
            vitals = _make_vitals(patient.id, hr=70.0 + (i % 10))
            alerts = api.ingest_vitals(vitals)
            assert isinstance(alerts, list)

    def test_100_readings_under_3_seconds(self) -> None:
        """100 readings complete in < 3 seconds."""
        api = PrecisionHealthAPI()
        patient = _make_patient()
        api.register_patient(patient)

        start = time.perf_counter()
        for i in range(100):
            vitals = _make_vitals(patient.id, hr=70.0 + (i % 10))
            api.ingest_vitals(vitals)
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, f"100 readings took {elapsed:.2f}s (limit: 3s)"

    def test_no_readings_dropped_under_load(self) -> None:
        """System doesn't drop readings under load."""
        pipeline = WearableDataPipeline()
        patient_id = "perf-patient-1"

        for i in range(100):
            reading = _make_reading(patient_id, 70.0 + (i % 10), float(i))
            pipeline.ingest(reading)

        window = pipeline.get_window(patient_id, "heart_rate")
        assert len(window) == 100, f"Expected 100 readings, got {len(window)}"

    def test_anomaly_detection_under_high_frequency(self) -> None:
        """Anomaly detection works correctly under high frequency."""
        pipeline = WearableDataPipeline()
        detector = AnomalyDetector()
        monitor = RemotePatientMonitor(pipeline, detector)
        patient_id = "perf-patient-1"

        # Ingest 95 normal readings
        for i in range(95):
            reading = _make_reading(patient_id, 70.0 + (i % 5), float(i))
            monitor.process_reading(reading)

        # Ingest 5 anomalous readings (spike to 200 bpm)
        alerts = []
        for i in range(5):
            reading = _make_reading(patient_id, 200.0, float(95 + i))
            alerts.extend(monitor.process_reading(reading))

        assert len(alerts) > 0, "No alerts generated for anomalous readings under high frequency"

    def test_critical_alerts_generated_in_batch(self) -> None:
        """Alerts are generated correctly for critical readings in the batch."""
        api = PrecisionHealthAPI()
        patient = _make_patient()
        api.register_patient(patient)

        # Ingest 95 normal readings
        for i in range(95):
            vitals = _make_vitals(patient.id, hr=70.0 + (i % 10))
            api.ingest_vitals(vitals)

        # Ingest 5 critical readings (HR > 150)
        for i in range(5):
            vitals = _make_vitals(patient.id, hr=180.0 + i)
            api.ingest_vitals(vitals)

        critical_alerts = api.get_active_alerts(patient.id, severity=AlertSeverity.CRITICAL)
        assert len(critical_alerts) > 0, "No critical alerts generated for critical readings"
        assert all(a.patient_id == patient.id for a in critical_alerts)
        assert all(a.severity == AlertSeverity.CRITICAL for a in critical_alerts)

    def test_audit_trail_intact_after_ingestion(self) -> None:
        """Audit trail remains intact after high-frequency ingestion."""
        api = PrecisionHealthAPI()
        patient = _make_patient()
        api.register_patient(patient)

        # Log audit events during ingestion
        for i in range(100):
            vitals = _make_vitals(patient.id, hr=70.0 + (i % 10))
            api.ingest_vitals(vitals)
            api.audit.log("perf-user", "ingest_vitals", "vitals", patient.id)

        assert api.audit.verify_chain(), "Audit chain broken after high-frequency ingestion"
        assert len(api.audit.get_events()) == 100
