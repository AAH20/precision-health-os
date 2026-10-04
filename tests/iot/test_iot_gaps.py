"""Tests for uncovered branches in precision_health_os.iot."""

from __future__ import annotations

from precision_health_os.iot import (
    AnomalyDetector,
    RemotePatientMonitor,
    SensorReading,
    WearableDataPipeline,
)


def _reading(patient_id: str, sensor_type: str, value: float, ts: float = 0.0) -> SensorReading:
    return SensorReading(
        device_id="dev1",
        patient_id=patient_id,
        sensor_type=sensor_type,
        value=value,
        timestamp=ts,
    )


class TestGetVitals:
    """Cover get_vitals for every sensor type (lines 74-77)."""

    def test_heart_rate(self):
        p = WearableDataPipeline()
        p.ingest(_reading("p1", "heart_rate", 72.0))
        v = p.get_vitals("p1")
        assert v.heart_rate_bpm == 72.0

    def test_spo2(self):
        p = WearableDataPipeline()
        p.ingest(_reading("p1", "spo2", 98.0))
        v = p.get_vitals("p1")
        assert v.spo2_percent == 98.0

    def test_blood_pressure(self):
        p = WearableDataPipeline()
        p.ingest(_reading("p1", "blood_pressure", 120.0))
        v = p.get_vitals("p1")
        assert v.blood_pressure_systolic == 120.0

    def test_temperature(self):
        p = WearableDataPipeline()
        p.ingest(_reading("p1", "temperature", 37.0))
        v = p.get_vitals("p1")
        assert v.temperature_celsius == 37.0

    def test_glucose(self):
        p = WearableDataPipeline()
        p.ingest(_reading("p1", "glucose", 95.0))
        v = p.get_vitals("p1")
        assert v.glucose_mg_dl == 95.0

    def test_all_sensors(self):
        p = WearableDataPipeline()
        p.ingest(_reading("p1", "heart_rate", 70.0))
        p.ingest(_reading("p1", "spo2", 97.0))
        p.ingest(_reading("p1", "blood_pressure", 118.0))
        p.ingest(_reading("p1", "temperature", 36.5))
        p.ingest(_reading("p1", "glucose", 90.0))
        v = p.get_vitals("p1")
        assert v.heart_rate_bpm == 70.0
        assert v.spo2_percent == 97.0
        assert v.blood_pressure_systolic == 118.0
        assert v.temperature_celsius == 36.5
        assert v.glucose_mg_dl == 90.0

    def test_no_readings_returns_empty_vitals(self):
        p = WearableDataPipeline()
        v = p.get_vitals("p1")
        assert v.heart_rate_bpm is None
        assert v.spo2_percent is None


class TestClearBuffer:
    """Cover clear_buffer with and without sensor_type (lines 86-88)."""

    def test_clear_specific_sensor(self):
        p = WearableDataPipeline()
        p.ingest(_reading("p1", "heart_rate", 70.0))
        p.ingest(_reading("p1", "spo2", 98.0))
        p.clear_buffer("p1", "heart_rate")
        assert p.get_window("p1", "heart_rate") == []
        assert len(p.get_window("p1", "spo2")) == 1

    def test_clear_all_sensors(self):
        p = WearableDataPipeline()
        p.ingest(_reading("p1", "heart_rate", 70.0))
        p.ingest(_reading("p1", "spo2", 98.0))
        p.ingest(_reading("p1", "temperature", 37.0))
        p.clear_buffer("p1")
        assert p.get_window("p1", "heart_rate") == []
        assert p.get_window("p1", "spo2") == []
        assert p.get_window("p1", "temperature") == []

    def test_clear_nonexistent_patient(self):
        p = WearableDataPipeline()
        p.clear_buffer("nonexistent")
        p.clear_buffer("nonexistent", "heart_rate")


class TestIqrEdgeCases:
    """Cover iqr len<4 branch (line 123) and iqr==0 path (line 136)."""

    def test_iqr_too_few_values(self):
        d = AnomalyDetector()
        results = d.detect_iqr([1.0, 2.0, 3.0])
        assert len(results) == 3
        assert all(not r.is_anomaly for r in results)

    def test_iqr_constant_values(self):
        d = AnomalyDetector()
        results = d.detect_iqr([72.5, 72.5, 72.5, 72.5])
        assert all(not r.is_anomaly for r in results)
        assert all(r.score == 0 for r in results)

    def test_iqr_zero_iqr_with_outlier(self):
        """A degenerate IQR must flag the outlier, not raise.

        Previously this raised ZeroDivisionError, taking the monitoring
        pipeline down for exactly the excursion that mattered.
        """
        d = AnomalyDetector()
        results = d.detect_iqr([70.0, 70.0, 70.0, 70.0, 120.0])
        assert results[-1].is_anomaly is True
        assert results[0].is_anomaly is False


class TestProcessReading:
    """Cover process_reading len<5 branch (line 231)."""

    def test_too_few_readings_returns_no_alerts(self):
        p = WearableDataPipeline()
        d = AnomalyDetector()
        m = RemotePatientMonitor(p, d)
        for i in range(4):
            alerts = m.process_reading(_reading("p1", "heart_rate", 70.0 + i, ts=float(i)))
            assert alerts == []

    def test_sufficient_readings_no_anomaly(self):
        p = WearableDataPipeline()
        d = AnomalyDetector()
        m = RemotePatientMonitor(p, d)
        for i in range(6):
            alerts = m.process_reading(_reading("p1", "heart_rate", 70.0, ts=float(i)))
            assert alerts == []


class TestGetPatientStatus:
    """Cover get_patient_status with insufficient data."""

    def test_no_data(self):
        p = WearableDataPipeline()
        d = AnomalyDetector()
        m = RemotePatientMonitor(p, d)
        status = m.get_patient_status("p1")
        assert status["patient_id"] == "p1"
        assert status["anomalies"] == {}

    def test_insufficient_data_for_anomaly_detection(self):
        p = WearableDataPipeline()
        d = AnomalyDetector()
        m = RemotePatientMonitor(p, d)
        for i in range(3):
            p.ingest(_reading("p1", "heart_rate", 70.0 + i, ts=float(i)))
        status = m.get_patient_status("p1")
        assert status["anomalies"] == {}


class TestEnsembleVote:
    """Cover ensemble vote tie-breaking."""

    def test_two_votes_triggers_anomaly(self):
        d = AnomalyDetector(z_threshold=0.5, iqr_multiplier=1000)
        vals = [48.0, 52.0, 50.0, 51.0, 49.0, 50.0, 52.0, 48.0, 50.0, 51.0, 49.0, 500.0]
        results = d.detect_ensemble(vals)
        assert results[-1].is_anomaly is True
        assert results[-1].details["votes"] == 2

    def test_one_vote_no_anomaly(self):
        d = AnomalyDetector(z_threshold=100, iqr_multiplier=1000)
        vals = [48.0, 52.0, 50.0, 51.0, 49.0, 50.0, 52.0, 48.0, 50.0, 51.0, 49.0, 500.0]
        results = d.detect_ensemble(vals)
        assert results[-1].is_anomaly is False
        assert results[-1].details["votes"] == 1


class TestEwmaSpikeRegression:
    """Regression: single spike must be caught by EWMA (self-inflation fix)."""

    def test_single_spike_detected(self):
        d = AnomalyDetector()
        vals = [50.0, 51.0, 49.0, 50.0, 52.0, 48.0, 50.0, 51.0, 49.0, 50.0, 500.0]
        results = d.detect_ewma(vals)
        assert results[-1].is_anomaly is True
        assert results[-1].score > 3.0

    def test_no_spike_no_anomaly(self):
        d = AnomalyDetector()
        vals = [50.0, 51.0, 49.0, 50.0, 52.0, 48.0, 50.0, 51.0, 49.0, 50.0]
        results = d.detect_ewma(vals)
        assert all(not r.is_anomaly for r in results)
