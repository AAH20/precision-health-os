"""Integration tests for vitals → anomaly detection → alert chain."""

from precision_health_os.clinical import AlertManager
from precision_health_os.iot import (
    AnomalyDetector,
    RemotePatientMonitor,
    SensorReading,
    WearableDataPipeline,
)
from precision_health_os.models import AlertSeverity


def _reading(patient_id, value, ts, sensor_type="heart_rate"):
    return SensorReading(
        device_id="dev-001",
        patient_id=patient_id,
        sensor_type=sensor_type,
        value=value,
        timestamp=ts,
        unit="bpm",
    )


def _monitor():
    pipeline = WearableDataPipeline()
    detector = AnomalyDetector()
    return pipeline, detector, RemotePatientMonitor(pipeline, detector)


def _ingest(monitor, patient_id, values):
    for i, v in enumerate(values):
        monitor.process_reading(_reading(patient_id, v, float(i)))


class TestNormalVitals:
    def test_normal_vitals_no_anomaly_no_alert(self):
        pipeline, detector, monitor = _monitor()
        alerts = []
        for i, v in enumerate([70, 72, 71, 73, 72]):
            alerts = monitor.process_reading(_reading("p1", v, float(i)))
        assert alerts == []
        values = [r.value for r in pipeline.get_window("p1", "heart_rate")]
        results = detector.detect_ensemble(values)
        assert not results[-1].is_anomaly


class TestSpikeDetection:
    def test_spike_triggers_anomaly_and_alert(self):
        _pipeline, _detector, monitor = _monitor()
        _ingest(monitor, "p1", [70, 72, 71, 73, 72])
        alerts = monitor.process_reading(_reading("p1", 200, 5.0))
        assert len(alerts) == 1
        assert alerts[0].patient_id == "p1"
        assert alerts[0].severity in (AlertSeverity.HIGH, AlertSeverity.MEDIUM)


class TestEnsembleDetection:
    def test_multiple_readings_ensemble_detection(self):
        pipeline, detector, monitor = _monitor()
        _ingest(monitor, "p1", [70, 72, 71, 73, 72, 74, 71, 73])
        window = pipeline.get_window("p1", "heart_rate")
        assert len(window) == 8
        results = detector.detect_ensemble([r.value for r in window])
        assert not results[-1].is_anomaly


class TestFullFlow:
    def test_vitals_anomaly_alert_acknowledge(self):
        _pipeline, _detector, monitor = _monitor()
        alert_manager = AlertManager()
        _ingest(monitor, "p1", [70, 72, 71, 73, 72])
        alerts = monitor.process_reading(_reading("p1", 200, 5.0))
        assert len(alerts) == 1
        alert = alert_manager.add_alert(alerts[0])
        assert alert.id == alerts[0].id
        active = alert_manager.get_active_alerts("p1")
        assert len(active) == 1
        result = alert_manager.acknowledge(alert.id, "dr_smith")
        assert result is not None
        assert result.acknowledged
        active = alert_manager.get_active_alerts("p1")
        assert len(active) == 0


class TestWarmupPeriod:
    def test_spike_during_warmup_not_detected(self):
        pipeline, detector, monitor = _monitor()
        _ingest(monitor, "p1", [200, 70, 72, 71, 73])
        values = [r.value for r in pipeline.get_window("p1", "heart_rate")]
        results = detector.detect_ensemble(values)
        assert not results[-1].is_anomaly

    def test_spike_after_warmup_detected(self):
        pipeline, detector, monitor = _monitor()
        _ingest(monitor, "p1", [70, 72, 71, 200, 73])
        values = [r.value for r in pipeline.get_window("p1", "heart_rate")]
        results = detector.detect_ensemble(values)
        assert results[3].is_anomaly


class TestEnsembleVoting:
    def test_two_of_three_votes_triggers_anomaly(self):
        detector = AnomalyDetector()
        values = [70, 72, 71, 73, 72, 200]
        results = detector.detect_ensemble(values)
        assert results[-1].is_anomaly
        assert results[-1].details["votes"] == 2
        assert results[-1].details["voters"] == 3

    def test_one_of_three_votes_no_anomaly(self):
        detector = AnomalyDetector()
        values = [70, 70, 70, 70, 70, 80]
        results = detector.detect_ensemble(values)
        assert not results[-1].is_anomaly
        assert results[-1].details["votes"] == 1
        assert results[-1].details["voters"] == 3
