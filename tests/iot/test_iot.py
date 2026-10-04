"""Tests for IoT module."""

from precision_health_os.iot import (
    AnomalyDetector,
    RemotePatientMonitor,
    SensorReading,
    WearableDataPipeline,
)


class TestWearableDataPipeline:
    """Tests for WearableDataPipeline."""

    def setup_method(self) -> None:
        self.pipeline = WearableDataPipeline(window_size=50)

    def test_ingest_reading(self) -> None:
        reading = SensorReading(
            device_id="dev001",
            patient_id="p001",
            sensor_type="heart_rate",
            value=72,
            timestamp=1000.0,
        )
        self.pipeline.ingest(reading)
        window = self.pipeline.get_window("p001", "heart_rate")
        assert len(window) == 1

    def test_window_size_limit(self) -> None:
        pipeline = WearableDataPipeline(window_size=5)
        for i in range(10):
            reading = SensorReading(
                device_id="dev001",
                patient_id="p001",
                sensor_type="heart_rate",
                value=70 + i,
                timestamp=float(i),
            )
            pipeline.ingest(reading)
        window = pipeline.get_window("p001", "heart_rate")
        assert len(window) == 5

    def test_get_vitals(self) -> None:
        for sensor_type, value in [
            ("heart_rate", 72),
            ("spo2", 98),
            ("blood_pressure", 120),
        ]:
            reading = SensorReading(
                device_id="dev001",
                patient_id="p001",
                sensor_type=sensor_type,
                value=value,
                timestamp=1000.0,
            )
            self.pipeline.ingest(reading)
        vitals = self.pipeline.get_vitals("p001")
        assert vitals.heart_rate_bpm == 72
        assert vitals.spo2_percent == 98
        assert vitals.blood_pressure_systolic == 120

    def test_clear_buffer(self) -> None:
        reading = SensorReading(
            device_id="dev001",
            patient_id="p001",
            sensor_type="heart_rate",
            value=72,
            timestamp=1000.0,
        )
        self.pipeline.ingest(reading)
        self.pipeline.clear_buffer("p001", "heart_rate")
        window = self.pipeline.get_window("p001", "heart_rate")
        assert len(window) == 0


class TestAnomalyDetector:
    """Tests for AnomalyDetector."""

    def setup_method(self) -> None:
        self.detector = AnomalyDetector(z_threshold=2.0)

    def test_zscore_normal_data(self) -> None:
        values = [10, 11, 9, 10, 10, 11, 9, 10]
        results = self.detector.detect_zscore(values)
        assert not any(r.is_anomaly for r in results)

    def test_zscore_anomaly(self) -> None:
        values = [10, 11, 9, 10, 10, 11, 9, 100]
        results = self.detector.detect_zscore(values)
        assert results[-1].is_anomaly

    def test_iqr_normal_data(self) -> None:
        values = [10, 11, 9, 10, 10, 11, 9, 10]
        results = self.detector.detect_iqr(values)
        assert not any(r.is_anomaly for r in results)

    def test_iqr_anomaly(self) -> None:
        values = [10, 11, 9, 10, 10, 11, 9, 100]
        results = self.detector.detect_iqr(values)
        assert results[-1].is_anomaly

    def test_ewma_detection(self) -> None:
        values = [10, 11, 9, 10, 10, 11, 9, 500]
        results = self.detector.detect_ewma(values, alpha=0.1, threshold=2.0)
        assert results[-1].is_anomaly

    def test_ensemble_detection(self) -> None:
        values = [10, 11, 9, 10, 10, 11, 9, 100]
        results = self.detector.detect_ensemble(values)
        assert results[-1].is_anomaly

    def test_insufficient_data(self) -> None:
        values = [10]
        results = self.detector.detect_zscore(values)
        assert not results[0].is_anomaly


class TestRemotePatientMonitor:
    """Tests for RemotePatientMonitor."""

    def setup_method(self) -> None:
        self.pipeline = WearableDataPipeline(window_size=50)
        self.detector = AnomalyDetector(z_threshold=2.0)
        self.monitor = RemotePatientMonitor(self.pipeline, self.detector)

    def test_process_normal_reading(self) -> None:
        # Seed with normal data
        for i in range(10):
            reading = SensorReading(
                device_id="dev001",
                patient_id="p001",
                sensor_type="heart_rate",
                value=70 + (i % 3),
                timestamp=float(i),
            )
            self.pipeline.ingest(reading)

        # Normal reading
        reading = SensorReading(
            device_id="dev001",
            patient_id="p001",
            sensor_type="heart_rate",
            value=72,
            timestamp=100.0,
        )
        alerts = self.monitor.process_reading(reading)
        assert len(alerts) == 0

    def test_process_anomalous_reading(self) -> None:
        # Seed with normal data
        for i in range(10):
            reading = SensorReading(
                device_id="dev001",
                patient_id="p001",
                sensor_type="heart_rate",
                value=70 + (i % 3),
                timestamp=float(i),
            )
            self.pipeline.ingest(reading)

        # Anomalous reading
        reading = SensorReading(
            device_id="dev001",
            patient_id="p001",
            sensor_type="heart_rate",
            value=150,
            timestamp=100.0,
        )
        alerts = self.monitor.process_reading(reading)
        assert len(alerts) >= 1

    def test_get_patient_status(self) -> None:
        for i in range(10):
            reading = SensorReading(
                device_id="dev001",
                patient_id="p001",
                sensor_type="heart_rate",
                value=70 + (i % 3),
                timestamp=float(i),
            )
            self.pipeline.ingest(reading)

        status = self.monitor.get_patient_status("p001")
        assert status["patient_id"] == "p001"
        assert "vitals" in status
        assert "anomalies" in status
