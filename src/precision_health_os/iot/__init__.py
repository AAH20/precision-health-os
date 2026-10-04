"""IoT module: Wearable data pipeline, remote monitoring, anomaly detection."""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from precision_health_os.models import AlertSeverity, ClinicalAlert, VitalSigns
from precision_health_os.utils import generate_id, z_score

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Single sensor reading."""

    device_id: str
    patient_id: str
    sensor_type: str
    value: float
    timestamp: float
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyResult:
    """Anomaly detection result."""

    is_anomaly: bool
    score: float
    method: str
    details: dict[str, Any] = field(default_factory=dict)


class WearableDataPipeline:
    """Real-time wearable data ingestion and processing pipeline."""

    def __init__(self, window_size: int = 100) -> None:
        """Initialize the wearable data pipeline."""
        self.window_size = window_size
        self._buffers: dict[str, deque[SensorReading]] = {}
        self._processors: list[Any] = []

    def ingest(self, reading: SensorReading) -> None:
        """Ingest a sensor reading."""
        key = f"{reading.patient_id}:{reading.sensor_type}"
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.window_size)
        self._buffers[key].append(reading)

    def get_window(self, patient_id: str, sensor_type: str) -> list[SensorReading]:
        """Get recent readings for a patient/sensor."""
        key = f"{patient_id}:{sensor_type}"
        return list(self._buffers.get(key, []))

    def get_vitals(self, patient_id: str) -> VitalSigns:
        """Aggregate recent readings into VitalSigns."""
        vitals = VitalSigns(patient_id=patient_id)
        for sensor_type in ["heart_rate", "spo2", "blood_pressure", "temperature", "glucose"]:
            readings = self.get_window(patient_id, sensor_type)
            if readings:
                latest = readings[-1]
                if sensor_type == "heart_rate":
                    vitals.heart_rate_bpm = latest.value
                elif sensor_type == "spo2":
                    vitals.spo2_percent = latest.value
                elif sensor_type == "blood_pressure":
                    vitals.blood_pressure_systolic = latest.value
                elif sensor_type == "temperature":
                    vitals.temperature_celsius = latest.value
                elif sensor_type == "glucose":
                    vitals.glucose_mg_dl = latest.value
        return vitals

    def clear_buffer(self, patient_id: str, sensor_type: str | None = None) -> None:
        """Clear buffer for a patient."""
        if sensor_type:
            key = f"{patient_id}:{sensor_type}"
            self._buffers.pop(key, None)
        else:
            keys_to_remove = [k for k in self._buffers if k.startswith(f"{patient_id}:")]
            for k in keys_to_remove:
                self._buffers.pop(k, None)


class AnomalyDetector:
    """Multi-method anomaly detection for wearable data."""

    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5) -> None:
        """Initialize the anomaly detector."""
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def detect_zscore(self, values: list[float]) -> list[AnomalyResult]:
        """Z-score based anomaly detection."""
        if len(values) < 3:
            return [AnomalyResult(False, 0.0, "zscore") for _ in values]

        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0

        results = []
        for v in values:
            z = z_score(v, mean, std)
            results.append(
                AnomalyResult(
                    is_anomaly=abs(z) > self.z_threshold,
                    score=abs(z),
                    method="zscore",
                    details={"z_score": z, "mean": mean, "std": std},
                )
            )
        return results

    def detect_iqr(self, values: list[float]) -> list[AnomalyResult]:
        """IQR-based anomaly detection."""
        if len(values) < 4:
            return [AnomalyResult(False, 0.0, "iqr") for _ in values]

        sorted_vals = sorted(values)
        q1 = sorted_vals[len(sorted_vals) // 4]
        q3 = sorted_vals[3 * len(sorted_vals) // 4]
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        results = []
        for v in values:
            is_anom = v < lower or v > upper
            score = max(
                0, (v - upper) / iqr if v > upper else (lower - v) / iqr if v < lower else 0
            )
            results.append(
                AnomalyResult(
                    is_anomaly=is_anom,
                    score=score,
                    method="iqr",
                    details={"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper},
                )
            )
        return results

    def detect_ewma(
        self, values: list[float], alpha: float = 0.3, threshold: float = 3.0
    ) -> list[AnomalyResult]:
        """Exponentially Weighted Moving Average anomaly detection.

        Each point is scored against the EWMA and variance estimated from
        *prior* observations only. Updating the variance with the current
        point's own deviation would let a large spike inflate its own
        denominator and escape detection.
        """
        if len(values) < 3:
            return [AnomalyResult(False, 0.0, "ewma") for _ in values]

        ewma = values[0]
        ewma_var = 0.0
        results = []

        for v in values:
            # Score against the model fitted on prior points.
            ewma_std = ewma_var**0.5 if ewma_var > 0 else 0
            diff = v - ewma
            score = abs(diff) / ewma_std if ewma_std > 0 else 0

            results.append(
                AnomalyResult(
                    is_anomaly=score > threshold,
                    score=score,
                    method="ewma",
                    details={"ewma": ewma, "ewma_std": ewma_std},
                )
            )

            # Then fold this observation into the model for the next point.
            ewma = alpha * v + (1 - alpha) * ewma
            ewma_var = alpha * diff**2 + (1 - alpha) * ewma_var

        return results

    def detect_ensemble(self, values: list[float]) -> list[AnomalyResult]:
        """Ensemble anomaly detection combining multiple methods."""
        z_results = self.detect_zscore(values)
        iqr_results = self.detect_iqr(values)
        ewma_results = self.detect_ewma(values)

        results = []
        for z, iqr, ewma in zip(z_results, iqr_results, ewma_results, strict=False):
            # Voting: 2 out of 3 methods must agree
            votes = sum([z.is_anomaly, iqr.is_anomaly, ewma.is_anomaly])
            avg_score = (z.score + iqr.score + ewma.score) / 3
            results.append(
                AnomalyResult(
                    is_anomaly=votes >= 2,
                    score=avg_score,
                    method="ensemble",
                    details={
                        "zscore": z.is_anomaly,
                        "iqr": iqr.is_anomaly,
                        "ewma": ewma.is_anomaly,
                        "votes": votes,
                    },
                )
            )
        return results


class RemotePatientMonitor:
    """Remote patient monitoring with alerting."""

    def __init__(self, pipeline: WearableDataPipeline, detector: AnomalyDetector) -> None:
        """Initialize the remote patient monitor."""
        self.pipeline = pipeline
        self.detector = detector
        self._alert_handlers: list[Any] = []

    def process_reading(self, reading: SensorReading) -> list[ClinicalAlert]:
        """Process a sensor reading and generate alerts if anomalous."""
        self.pipeline.ingest(reading)

        # Get recent window
        window = self.pipeline.get_window(reading.patient_id, reading.sensor_type)
        values = [r.value for r in window]

        if len(values) < 5:
            return []

        # Detect anomalies
        results = self.detector.detect_ensemble(values)
        latest = results[-1]

        if not latest.is_anomaly:
            return []

        # Generate alert
        severity = AlertSeverity.HIGH if latest.score > 5 else AlertSeverity.MEDIUM
        alert = ClinicalAlert(
            id=generate_id(),
            patient_id=reading.patient_id,
            severity=severity,
            title=f"Anomalous {reading.sensor_type.replace('_', ' ').title()}",
            description=(
                f"Detected anomalous {reading.sensor_type} = {reading.value:.1f}{reading.unit} "
                f"(score: {latest.score:.2f}, method: {latest.method})"
            ),
            source=f"anomaly:{latest.method}",
            confidence=min(latest.score / 10, 0.99),
            recommendations=[
                "Review patient status",
                "Consider clinical assessment",
                "Verify sensor placement",
            ],
        )

        return [alert]

    def get_patient_status(self, patient_id: str) -> dict[str, Any]:
        """Get comprehensive patient status."""
        vitals = self.pipeline.get_vitals(patient_id)
        status: dict[str, Any] = {"patient_id": patient_id, "vitals": vitals.model_dump()}

        # Check each sensor for anomalies
        anomalies: dict[str, Any] = {}
        for sensor_type in ["heart_rate", "spo2", "blood_pressure", "temperature", "glucose"]:
            window = self.pipeline.get_window(patient_id, sensor_type)
            if len(window) >= 5:
                values = [r.value for r in window]
                results = self.detector.detect_ensemble(values)
                anomalies[sensor_type] = {
                    "latest_score": results[-1].score,
                    "is_anomaly": results[-1].is_anomaly,
                    "method": results[-1].method,
                }
        status["anomalies"] = anomalies
        return status
