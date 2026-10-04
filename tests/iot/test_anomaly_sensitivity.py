"""Regression tests for anomaly detection sensitivity.

Regression 1: EWMA scored each point against a variance estimate that already
included that point's own deviation, so a single large spike inflated its own
denominator and went undetected.

Regression 2: with only the IQR arm firing, the 2-of-3 ensemble failed to flag
a clinically significant single-point excursion (a 2.9x jump in heart rate).
"""

import pytest

from precision_health_os.iot import (
    AnomalyDetector,
    RemotePatientMonitor,
    SensorReading,
    WearableDataPipeline,
)


class TestEWMANoSelfInflation:
    """EWMA must test a point against variance from *prior* observations."""

    def test_single_spike_is_flagged(self) -> None:
        """A large single-point spike must be detected by EWMA alone."""
        detector = AnomalyDetector()
        values = [70, 72, 71, 70, 73, 71, 72, 71, 200]
        results = detector.detect_ewma(values)
        assert results[-1].is_anomaly, "EWMA failed to flag a 2.9x single-point spike"

    def test_steady_signal_is_not_flagged(self) -> None:
        detector = AnomalyDetector()
        values = [70, 72, 71, 70, 73, 71, 72, 71, 72]
        results = detector.detect_ewma(values)
        assert not any(r.is_anomaly for r in results)

    def test_score_is_not_self_normalized(self) -> None:
        """The spike's score must exceed the threshold by a wide margin."""
        detector = AnomalyDetector()
        results = detector.detect_ewma([70, 72, 71, 70, 73, 71, 72, 71, 200])
        assert results[-1].score > 5.0, "spike score collapsed — variance self-inflated"


class TestEnsembleCatchesClinicallySignificantSpike:
    """The production ensemble must flag a single large excursion."""

    def test_heart_rate_spike_flagged(self) -> None:
        """70s baseline -> 200 bpm is a medical emergency, not noise."""
        detector = AnomalyDetector()
        results = detector.detect_ensemble([70, 72, 71, 70, 73, 71, 72, 71, 200])
        assert results[-1].is_anomaly
        assert results[-1].details["votes"] >= 2

    def test_normal_variation_not_flagged(self) -> None:
        detector = AnomalyDetector()
        results = detector.detect_ensemble([70, 72, 71, 70, 73, 71, 72, 71, 72])
        assert not any(r.is_anomaly for r in results)

    def test_remote_monitor_raises_alert_on_spike(self) -> None:
        """End-to-end: the monitor must emit a ClinicalAlert for the spike."""
        pipeline = WearableDataPipeline(window_size=50)
        monitor = RemotePatientMonitor(pipeline, AnomalyDetector())
        for i, v in enumerate([70, 72, 71, 70, 73, 71, 72, 71]):
            pipeline.ingest(
                SensorReading(
                    device_id="d1",
                    patient_id="p1",
                    sensor_type="heart_rate",
                    value=v,
                    timestamp=float(i),
                )
            )
        alerts = monitor.process_reading(
            SensorReading(
                device_id="d1",
                patient_id="p1",
                sensor_type="heart_rate",
                value=200,
                timestamp=100.0,
            )
        )
        assert len(alerts) >= 1
        assert alerts[0].severity.value in ("high", "critical")


class TestNoRegressionOnExistingBehaviour:
    """Pre-existing detections must keep working."""

    @pytest.mark.parametrize("value", [100, 500])
    def test_original_spike_still_detected(self, value: int) -> None:
        detector = AnomalyDetector(z_threshold=2.0)
        results = detector.detect_ensemble([10, 11, 9, 10, 10, 11, 9, value])
        assert results[-1].is_anomaly

    def test_short_series_is_safe(self) -> None:
        detector = AnomalyDetector()
        assert not any(r.is_anomaly for r in detector.detect_ewma([70, 71]))
        assert not any(r.is_anomaly for r in detector.detect_zscore([70]))
