"""Tests for precision_health_os.iot module — covering uncovered branches."""

from precision_health_os.iot import SensorReading, WearableDataPipeline


def test_get_vitals_glucose_branch():
    """Cover the glucose branch (line 76->64) in get_vitals()."""
    pipeline = WearableDataPipeline()
    reading = SensorReading(
        device_id="dev-1",
        patient_id="pat-1",
        sensor_type="glucose",
        value=142.0,
        timestamp=1000.0,
        unit="mg/dL",
    )
    pipeline.ingest(reading)
    vitals = pipeline.get_vitals("pat-1")
    assert vitals.glucose_mg_dl == 142.0
