"""Performance tests for concurrent patient registration and vitals ingestion."""

import threading
import time
from datetime import UTC, datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.models import Patient, VitalSigns


def _make_patient(i: int) -> Patient:
    return Patient(
        id=f"P{i:04d}",
        mrn=f"MRN{i:06d}",
        name=f"Patient {i}",
        date_of_birth=datetime(1990, 1, 1, tzinfo=UTC),
        sex="male" if i % 2 == 0 else "female",
    )


def _make_vitals(patient_id: str) -> VitalSigns:
    return VitalSigns(
        patient_id=patient_id,
        heart_rate_bpm=72.0,
        blood_pressure_systolic=120.0,
        blood_pressure_diastolic=80.0,
        spo2_percent=98.0,
        temperature_celsius=37.0,
        respiratory_rate=16.0,
        glucose_mg_dl=95.0,
    )


class TestConcurrentPatientRegistration:
    def test_register_50_patients_concurrently(self):
        api = PrecisionHealthAPI()
        patients = [_make_patient(i) for i in range(50)]
        results = [None] * 50
        errors = [None] * 50

        def register(idx, patient):
            try:
                results[idx] = api.register_patient(patient)
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=register, args=(i, p)) for i, p in enumerate(patients)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        assert all(e is None for e in errors), f"Errors: {[e for e in errors if e]}"
        assert all(r is not None for r in results)
        assert len(results) == 50
        assert elapsed < 5.0, f"Registration took {elapsed:.2f}s, expected < 5s"

    def test_concurrent_registration_no_data_corruption(self):
        api = PrecisionHealthAPI()
        patients = [_make_patient(i) for i in range(50)]

        threads = [threading.Thread(target=api.register_patient, args=(p,)) for p in patients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for p in patients:
            stored = api.get_patient(p.id)
            assert stored is not None
            assert stored.id == p.id
            assert stored.mrn == p.mrn
            assert stored.name == p.name

    def test_system_consistent_after_concurrent_registration(self):
        api = PrecisionHealthAPI()
        patients = [_make_patient(i) for i in range(50)]

        threads = [threading.Thread(target=api.register_patient, args=(p,)) for p in patients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        status = api.get_system_status()
        assert status["patients_registered"] == 50


class TestConcurrentVitalsIngestion:
    def test_ingest_vitals_for_50_patients_concurrently(self):
        api = PrecisionHealthAPI()
        patients = [_make_patient(i) for i in range(50)]
        for p in patients:
            api.register_patient(p)

        results = [None] * 50
        errors = [None] * 50

        def ingest(idx, patient):
            try:
                vitals = _make_vitals(patient.id)
                results[idx] = api.ingest_vitals(vitals)
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=ingest, args=(i, p)) for i, p in enumerate(patients)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        assert all(e is None for e in errors), f"Errors: {[e for e in errors if e]}"
        assert all(r is not None for r in results)
        assert elapsed < 5.0, f"Vitals ingestion took {elapsed:.2f}s, expected < 5s"

    def test_concurrent_vitals_no_data_corruption(self):
        api = PrecisionHealthAPI()
        patients = [_make_patient(i) for i in range(50)]
        for p in patients:
            api.register_patient(p)

        def ingest(patient):
            vitals = _make_vitals(patient.id)
            api.ingest_vitals(vitals)

        threads = [threading.Thread(target=ingest, args=(p,)) for p in patients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for p in patients:
            stored = api.get_patient(p.id)
            assert stored is not None
            assert stored.id == p.id

    def test_system_consistent_after_concurrent_vitals(self):
        api = PrecisionHealthAPI()
        patients = [_make_patient(i) for i in range(50)]
        for p in patients:
            api.register_patient(p)

        def ingest(patient):
            vitals = _make_vitals(patient.id)
            api.ingest_vitals(vitals)

        threads = [threading.Thread(target=ingest, args=(p,)) for p in patients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        status = api.get_system_status()
        assert status["patients_registered"] == 50
