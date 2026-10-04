"""Tests for the quality evolution ledger.

Written BEFORE implementation (TDD RED phase).

The ledger persists quality metrics across runs so improvement is a measured
trend, not a claim, and a regression in any tracked metric fails the gate.
"""

from __future__ import annotations

import json

import pytest

from precision_health_os.evolution import (
    TRACKED_METRICS,
    QualityLedger,
    QualitySnapshot,
)


class TestQualitySnapshot:
    """A single point-in-time measurement of the project's quality."""

    def test_create_snapshot(self) -> None:
        snap = QualitySnapshot(
            label="run-1",
            metrics={"tests": 516, "coverage": 98.57, "benchmark_pass_rate": 1.0},
        )
        assert snap.label == "run-1"
        assert snap.metrics["tests"] == 516

    def test_snapshot_to_dict_is_serializable(self) -> None:
        snap = QualitySnapshot(label="r", metrics={"tests": 10})
        payload = json.dumps(snap.to_dict())
        assert json.loads(payload)["label"] == "r"

    def test_snapshot_from_dict_roundtrip(self) -> None:
        snap = QualitySnapshot(label="r", metrics={"tests": 10, "coverage": 90.0})
        restored = QualitySnapshot.from_dict(snap.to_dict())
        assert restored.label == snap.label
        assert restored.metrics == snap.metrics

    def test_snapshot_requires_known_metrics(self) -> None:
        """An unknown metric name is a typo, not silently accepted."""
        with pytest.raises(ValueError, match="unknown metric"):
            QualitySnapshot(label="r", metrics={"not_a_metric": 1.0})


class TestQualityLedger:
    """Append-only history of quality snapshots."""

    def test_empty_ledger(self) -> None:
        ledger = QualityLedger()
        assert ledger.snapshots() == []
        assert ledger.latest() is None
        assert ledger.baseline() is None

    def test_record_and_retrieve(self) -> None:
        ledger = QualityLedger()
        ledger.record(QualitySnapshot(label="a", metrics={"tests": 100}))
        ledger.record(QualitySnapshot(label="b", metrics={"tests": 200}))
        assert len(ledger.snapshots()) == 2
        assert ledger.latest().label == "b"
        assert ledger.baseline().label == "a"

    def test_ledger_survives_roundtrip_through_file(self, tmp_path) -> None:
        path = tmp_path / "ledger.json"
        ledger = QualityLedger(path=path)
        ledger.record(QualitySnapshot(label="a", metrics={"tests": 100}))
        ledger.save()

        reloaded = QualityLedger(path=path)
        assert len(reloaded.snapshots()) == 1
        assert reloaded.latest().metrics["tests"] == 100

    def test_missing_file_yields_empty_ledger(self, tmp_path) -> None:
        ledger = QualityLedger(path=tmp_path / "does-not-exist.json")
        assert ledger.snapshots() == []

    def test_corrupt_file_does_not_crash(self, tmp_path) -> None:
        """A corrupt ledger must not take the gate down."""
        path = tmp_path / "ledger.json"
        path.write_text("{not valid json")
        ledger = QualityLedger(path=path)
        assert ledger.snapshots() == []

    def test_unknown_metric_in_stored_file_is_skipped(self, tmp_path) -> None:
        """Forward compatibility: a metric removed later must not break load."""
        path = tmp_path / "ledger.json"
        path.write_text(
            json.dumps(
                {
                    "snapshots": [
                        {"label": "old", "metrics": {"tests": 5, "gone_metric": 1.0}},
                        {"label": "new", "metrics": {"tests": 9}},
                    ]
                }
            )
        )
        ledger = QualityLedger(path=path)
        assert len(ledger.snapshots()) == 2
        assert "gone_metric" not in ledger.snapshots()[0].metrics
        assert ledger.snapshots()[0].metrics["tests"] == 5


class TestRegressionDetection:
    """Comparing a new snapshot against the best previously recorded."""

    def test_no_regression_when_improving(self) -> None:
        ledger = QualityLedger()
        ledger.record(QualitySnapshot(label="a", metrics={"tests": 100, "coverage": 90.0}))
        regressions = ledger.regressions(
            QualitySnapshot(label="b", metrics={"tests": 200, "coverage": 95.0})
        )
        assert regressions == []

    def test_regression_detected_on_drop(self) -> None:
        ledger = QualityLedger()
        ledger.record(QualitySnapshot(label="a", metrics={"tests": 500, "coverage": 98.0}))
        regressions = ledger.regressions(
            QualitySnapshot(label="b", metrics={"tests": 400, "coverage": 98.0})
        )
        assert len(regressions) == 1
        assert regressions[0]["metric"] == "tests"
        assert regressions[0]["previous"] == 500
        assert regressions[0]["current"] == 400

    def test_regression_reports_every_dropped_metric(self) -> None:
        ledger = QualityLedger()
        ledger.record(
            QualitySnapshot(
                label="a",
                metrics={"tests": 500, "coverage": 98.0, "benchmark_pass_rate": 1.0},
            )
        )
        regressions = ledger.regressions(
            QualitySnapshot(
                label="b",
                metrics={"tests": 400, "coverage": 90.0, "benchmark_pass_rate": 0.8},
            )
        )
        assert {r["metric"] for r in regressions} == {
            "tests",
            "coverage",
            "benchmark_pass_rate",
        }

    def test_equality_is_not_a_regression(self) -> None:
        """Holding steady is fine; only a drop counts."""
        ledger = QualityLedger()
        ledger.record(QualitySnapshot(label="a", metrics={"tests": 500, "coverage": 98.0}))
        assert (
            ledger.regressions(QualitySnapshot(label="b", metrics={"tests": 500, "coverage": 98.0}))
            == []
        )

    def test_first_snapshot_cannot_regress(self) -> None:
        ledger = QualityLedger()
        assert (
            ledger.regressions(
                QualitySnapshot(label="first", metrics={"tests": 1, "coverage": 1.0})
            )
            == []
        )

    def test_missing_metric_in_new_snapshot_is_not_a_regression(self) -> None:
        """Only compare metrics present in both."""
        ledger = QualityLedger()
        ledger.record(QualitySnapshot(label="a", metrics={"tests": 100, "coverage": 90.0}))
        assert ledger.regressions(QualitySnapshot(label="b", metrics={"tests": 100})) == []


class TestTrend:
    """Direction of travel across the whole history."""

    def test_improving_trend(self) -> None:
        ledger = QualityLedger()
        for i, cov in enumerate([80.0, 90.0, 98.0], start=1):
            ledger.record(QualitySnapshot(label=f"r{i}", metrics={"coverage": cov}))
        assert ledger.trend("coverage") == "improving"

    def test_declining_trend(self) -> None:
        ledger = QualityLedger()
        for i, cov in enumerate([98.0, 90.0, 80.0], start=1):
            ledger.record(QualitySnapshot(label=f"r{i}", metrics={"coverage": cov}))
        assert ledger.trend("coverage") == "declining"

    def test_flat_trend(self) -> None:
        ledger = QualityLedger()
        for i in range(3):
            ledger.record(QualitySnapshot(label=f"r{i}", metrics={"coverage": 95.0}))
        assert ledger.trend("coverage") == "flat"

    def test_trend_with_insufficient_history(self) -> None:
        ledger = QualityLedger()
        assert ledger.trend("coverage") == "unknown"
        ledger.record(QualitySnapshot(label="one", metrics={"coverage": 95.0}))
        assert ledger.trend("coverage") == "unknown"

    def test_trend_rejects_unknown_metric(self) -> None:
        ledger = QualityLedger()
        with pytest.raises(ValueError, match="unknown metric"):
            ledger.trend("bogus")


class TestTrackedMetricsContract:
    """The tracked metric set must be internally consistent."""

    def test_tracked_metrics_is_non_empty(self) -> None:
        assert len(TRACKED_METRICS) >= 3

    def test_every_tracked_metric_declares_direction(self) -> None:
        """Each metric must say whether higher or lower is better."""
        for name, spec in TRACKED_METRICS.items():
            assert "higher_is_better" in spec, f"{name} missing direction"
            assert isinstance(spec["higher_is_better"], bool)

    def test_core_metrics_are_tracked(self) -> None:
        """The gate depends on these specifically."""
        for required in ("tests", "coverage", "benchmark_pass_rate"):
            assert required in TRACKED_METRICS


class TestLedgerSummary:
    """A compact human-readable report."""

    def test_summary_shape(self) -> None:
        ledger = QualityLedger()
        ledger.record(QualitySnapshot(label="a", metrics={"tests": 100, "coverage": 80.0}))
        ledger.record(QualitySnapshot(label="b", metrics={"tests": 200, "coverage": 90.0}))
        summary = ledger.summary()
        assert summary["snapshots"] == 2
        assert summary["latest_label"] == "b"
        assert summary["baseline_label"] == "a"
        assert "coverage" in summary["trends"]
        assert summary["trends"]["coverage"] == "improving"

    def test_summary_on_empty_ledger(self) -> None:
        summary = QualityLedger().summary()
        assert summary["snapshots"] == 0
        assert summary["latest_label"] is None
        assert summary["trends"] == {}
