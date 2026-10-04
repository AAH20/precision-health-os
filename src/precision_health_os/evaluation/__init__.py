"""Evaluation & evolution framework.

Provides measurable quality gates (`Benchmark` / `EvaluationHarness`) and
improvement tracking across generations (`EvolutionTracker`), so the platform
can be scored against explicit thresholds rather than subjective judgement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Metric:
    """A single measured quantity with a direction of goodness.

    Attributes:
        name: Metric identifier, e.g. ``"accuracy"``.
        value: Measured value.
        unit: Unit of measure, e.g. ``"ratio"``, ``"ms"``.
        higher_is_better: True when a larger value is an improvement.
    """

    name: str
    value: float
    unit: str = ""
    higher_is_better: bool = True

    def better_than(self, other: Metric) -> bool:
        """Return True when this metric is a strict improvement on ``other``."""
        if self.higher_is_better:
            return self.value > other.value
        return self.value < other.value

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
        }


@dataclass
class BenchmarkResult:
    """Outcome of running one benchmark."""

    benchmark_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    score: float = 0.0
    missing_metrics: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "benchmark_name": self.benchmark_name,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "passed": self.passed,
            "score": self.score,
            "missing_metrics": self.missing_metrics,
            "error": self.error,
        }


class Benchmark:
    """A named, runnable measurement checked against pass thresholds.

    Args:
        name: Benchmark identifier.
        fn: Callable returning a ``{metric_name: value}`` mapping.
        thresholds: Required value per metric. For metrics listed in
            ``lower_is_better`` the measured value must be at or below the
            threshold; otherwise it must be at or above it.
        lower_is_better: Metric names where smaller is better (e.g. latency).
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[], dict[str, float]],
        thresholds: dict[str, float],
        lower_is_better: set[str] | None = None,
    ) -> None:
        """Initialize the benchmark."""
        self.name = name
        self._fn = fn
        self.thresholds = dict(thresholds)
        self._lower_is_better = set(lower_is_better or ())

    def run(self) -> BenchmarkResult:
        """Execute the benchmark and evaluate it against its thresholds.

        A benchmark that raises, or that fails to report a required metric, is
        recorded as failed rather than propagated — one broken benchmark must
        not abort a suite.
        """
        try:
            measured = self._fn()
        except Exception as exc:
            return BenchmarkResult(
                benchmark_name=self.name,
                thresholds=dict(self.thresholds),
                passed=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        metrics = {k: float(v) for k, v in measured.items()}
        missing = [k for k in self.thresholds if k not in metrics]

        if missing:
            return BenchmarkResult(
                benchmark_name=self.name,
                metrics=metrics,
                thresholds=dict(self.thresholds),
                passed=False,
                score=0.0,
                missing_metrics=missing,
            )

        satisfied = 0
        for key, required in self.thresholds.items():
            value = metrics[key]
            ok = value <= required if key in self._lower_is_better else value >= required
            if ok:
                satisfied += 1

        total = len(self.thresholds)
        score = satisfied / total if total else 1.0

        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            thresholds=dict(self.thresholds),
            passed=satisfied == total,
            score=score,
        )


class EvaluationHarness:
    """Registers benchmarks and runs them as an ordered suite."""

    def __init__(self) -> None:
        """Initialize an empty suite."""
        self._benchmarks: list[Benchmark] = []

    def register(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite."""
        self._benchmarks.append(benchmark)

    def run_all(self) -> list[BenchmarkResult]:
        """Run every registered benchmark in registration order."""
        return [b.run() for b in self._benchmarks]

    def summary(self) -> dict[str, Any]:
        """Return aggregate pass statistics for the suite."""
        results = self.run_all()
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total else 0.0,
            "mean_score": (sum(r.score for r in results) / total) if total else 0.0,
            "results": [r.to_dict() for r in results],
        }


class EvolutionTracker:
    """Tracks a metric's movement across generations of improvement.

    Args:
        metric: Name of the metric being tracked.
    """

    def __init__(self, metric: str) -> None:
        """Initialize the tracker for a named metric."""
        self.metric = metric
        self._history: list[dict[str, Any]] = []

    def record(self, generation: int, score: float) -> None:
        """Record a measurement for a generation."""
        self._history.append({"generation": generation, "score": float(score)})

    def history(self) -> list[dict[str, Any]]:
        """Return the recorded history in insertion order."""
        return list(self._history)

    def best_generation(self) -> int | None:
        """Return the generation with the highest score, or None if empty."""
        if not self._history:
            return None
        return max(self._history, key=lambda h: h["score"])["generation"]

    def best_score(self) -> float:
        """Return the highest score recorded, or 0.0 if empty."""
        if not self._history:
            return 0.0
        return max(h["score"] for h in self._history)

    def improvement(self) -> float:
        """Return the net change from first to last recorded score."""
        if len(self._history) < 2:
            return 0.0
        return self._history[-1]["score"] - self._history[0]["score"]

    def is_improving(self) -> bool:
        """Return True when the sequence of scores is monotonically increasing."""
        if len(self._history) < 2:
            return False
        scores = [h["score"] for h in self._history]
        return all(b > a for a, b in pairwise(scores))

    def regressed(self) -> bool:
        """Return True when the latest score is below the best recorded."""
        if len(self._history) < 2:
            return False
        return self._history[-1]["score"] < self.best_score()
