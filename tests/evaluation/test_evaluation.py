"""Tests for the evaluation & evolution framework.

Written BEFORE implementation (TDD RED phase).
"""

from __future__ import annotations

import pytest

from precision_health_os.evaluation import (
    Benchmark,
    BenchmarkResult,
    EvaluationHarness,
    EvolutionTracker,
    Metric,
)


class TestMetric:
    """A single measured quantity with a direction of goodness."""

    def test_create_metric(self) -> None:
        m = Metric(name="accuracy", value=0.95, unit="ratio", higher_is_better=True)
        assert m.name == "accuracy"
        assert m.value == 0.95
        assert m.unit == "ratio"

    def test_higher_is_better_direction(self) -> None:
        fast = Metric("accuracy", 0.90, "ratio", True)
        slow = Metric("accuracy", 0.80, "ratio", True)
        assert fast.better_than(slow)
        assert not slow.better_than(fast)

    def test_lower_is_better_direction(self) -> None:
        quick = Metric("latency", 10.0, "ms", False)
        sluggish = Metric("latency", 20.0, "ms", False)
        assert quick.better_than(sluggish)
        assert not sluggish.better_than(quick)

    def test_equal_values_are_not_better(self) -> None:
        a = Metric("accuracy", 0.9, "ratio", True)
        b = Metric("accuracy", 0.9, "ratio", True)
        assert not a.better_than(b)

    def test_to_dict_is_serializable(self) -> None:
        d = Metric("accuracy", 0.9, "ratio", True).to_dict()
        assert d == {
            "name": "accuracy",
            "value": 0.9,
            "unit": "ratio",
            "higher_is_better": True,
        }


class TestBenchmark:
    """A named, runnable measurement with pass thresholds."""

    def test_run_produces_result(self) -> None:
        bench = Benchmark(
            name="addition",
            fn=lambda: {"correctness": 1.0},
            thresholds={"correctness": 1.0},
        )
        result = bench.run()
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "addition"
        assert result.passed is True

    def test_fails_when_below_threshold(self) -> None:
        bench = Benchmark(
            name="strict",
            fn=lambda: {"correctness": 0.5},
            thresholds={"correctness": 1.0},
        )
        assert bench.run().passed is False

    def test_respects_lower_is_better_threshold(self) -> None:
        """A latency budget passes when measured latency is BELOW it."""
        bench = Benchmark(
            name="fast",
            fn=lambda: {"latency": 5.0},
            thresholds={"latency": 10.0},
            lower_is_better={"latency"},
        )
        assert bench.run().passed is True

    def test_lower_is_better_fails_when_exceeded(self) -> None:
        bench = Benchmark(
            name="slow",
            fn=lambda: {"latency": 50.0},
            thresholds={"latency": 10.0},
            lower_is_better={"latency"},
        )
        assert bench.run().passed is False

    def test_missing_metric_fails_rather_than_silently_passing(self) -> None:
        """A benchmark that does not report a required metric must fail."""
        bench = Benchmark(
            name="incomplete",
            fn=lambda: {"other": 1.0},
            thresholds={"correctness": 1.0},
        )
        result = bench.run()
        assert result.passed is False
        assert "correctness" in result.missing_metrics

    def test_exception_does_not_abort_the_harness(self) -> None:
        """A raising benchmark is recorded as failed, not propagated."""

        def boom() -> dict[str, float]:
            raise RuntimeError("solver exploded")

        result = Benchmark(name="boom", fn=boom, thresholds={}).run()
        assert result.passed is False
        assert "solver exploded" in (result.error or "")

    def test_score_is_bounded(self) -> None:
        bench = Benchmark(
            name="scored",
            fn=lambda: {"a": 1.0, "b": 1.0},
            thresholds={"a": 1.0, "b": 1.0},
        )
        score = bench.run().score
        assert 0.0 <= score <= 1.0


class TestEvaluationHarness:
    """Registers benchmarks and runs them as a suite."""

    def test_register_and_run_all(self) -> None:
        harness = EvaluationHarness()
        harness.register(Benchmark("one", lambda: {"x": 1.0}, {"x": 1.0}))
        harness.register(Benchmark("two", lambda: {"y": 2.0}, {"y": 2.0}))
        results = harness.run_all()
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_summary_reports_pass_rate(self) -> None:
        harness = EvaluationHarness()
        harness.register(Benchmark("good", lambda: {"x": 1.0}, {"x": 1.0}))
        harness.register(Benchmark("bad", lambda: {"x": 0.0}, {"x": 1.0}))
        summary = harness.summary()
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["pass_rate"] == 0.5

    def test_run_all_is_deterministic_in_order(self) -> None:
        harness = EvaluationHarness()
        for n in ("a", "b", "c"):
            harness.register(Benchmark(n, lambda: {"x": 1.0}, {"x": 1.0}))
        names = [r.benchmark_name for r in harness.run_all()]
        assert names == ["a", "b", "c"]

    def test_empty_harness_summary(self) -> None:
        summary = EvaluationHarness().summary()
        assert summary["total"] == 0
        assert summary["pass_rate"] == 0.0


class TestEvolutionTracker:
    """Tracks metric movement across generations of improvement."""

    def test_record_and_best(self) -> None:
        tracker = EvolutionTracker(metric="score")
        tracker.record(generation=1, score=0.60)
        tracker.record(generation=2, score=0.85)
        assert tracker.best_generation() == 2
        assert tracker.best_score() == 0.85

    def test_improvement_across_generations(self) -> None:
        tracker = EvolutionTracker(metric="score")
        tracker.record(1, 0.50)
        tracker.record(2, 0.75)
        assert tracker.improvement() == pytest.approx(0.25)

    def test_is_improving_true_for_monotonic_gain(self) -> None:
        tracker = EvolutionTracker(metric="score")
        for gen, score in enumerate([0.40, 0.55, 0.70, 0.90], start=1):
            tracker.record(gen, score)
        assert tracker.is_improving() is True

    def test_is_improving_false_when_regressing(self) -> None:
        tracker = EvolutionTracker(metric="score")
        for gen, score in enumerate([0.90, 0.70, 0.50], start=1):
            tracker.record(gen, score)
        assert tracker.is_improving() is False

    def test_empty_tracker_is_safe(self) -> None:
        tracker = EvolutionTracker(metric="score")
        assert tracker.best_generation() is None
        assert tracker.best_score() == 0.0
        assert tracker.improvement() == 0.0
        assert tracker.is_improving() is False

    def test_history_is_serializable(self) -> None:
        tracker = EvolutionTracker(metric="score")
        tracker.record(1, 0.5)
        tracker.record(2, 0.7)
        history = tracker.history()
        assert history == [
            {"generation": 1, "score": 0.5},
            {"generation": 2, "score": 0.7},
        ]

    def test_regression_is_detected(self) -> None:
        tracker = EvolutionTracker(metric="score")
        tracker.record(1, 0.90)
        tracker.record(2, 0.60)
        assert tracker.regressed() is True
        assert tracker.best_score() == 0.90
