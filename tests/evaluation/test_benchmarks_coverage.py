"""Tests for benchmarks.py coverage of uncovered branches."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from precision_health_os import benchmarks
from precision_health_os.evaluation import Benchmark, EvaluationHarness
from precision_health_os.iot import AnomalyDetector
from precision_health_os.optimization import (
    KnapsackSolver,
    TSPSolver,
    VRPTimeWindowsSolver,
)


def test_tsp_quality_worse_than_nn():
    """Cover line 88: TSP solver worse than nearest-neighbour."""

    def bad_solve(self, pts):
        return SimpleNamespace(locations=list(reversed(pts)))

    with patch.object(TSPSolver, "solve", bad_solve):
        result = benchmarks.bench_tsp_quality()

    assert result["no_regression_vs_nn"] < 1.0


def test_tsp_quality_zero_optimal():
    """Cover line 102->94: optimal length is zero."""
    with patch.object(benchmarks, "_exact_tsp_len", return_value=0.0):
        result = benchmarks.bench_tsp_quality()

    assert result["within_10pct_of_optimal"] == 0.0


def test_knapsack_suboptimal():
    """Cover line 139->119: knapsack doesn't match brute-force optimum."""

    def bad_solve(self, items):
        return []

    with patch.object(KnapsackSolver, "solve", bad_solve):
        result = benchmarks.bench_knapsack_optimality()

    assert result["optimality_rate"] < 1.0


def test_vrp_incomplete():
    """Cover line 168->170: VRP doesn't serve all nodes."""

    def bad_solve(self, locs):
        return []

    with patch.object(VRPTimeWindowsSolver, "solve", bad_solve):
        result = benchmarks.bench_vrp_feasibility()

    assert result["all_nodes_served_rate"] == 0.0


def test_vrp_capacity_breach():
    """Cover line 170->152: VRP breaches vehicle capacity."""

    def bad_solve(self, locs):
        return [SimpleNamespace(locations=locs)]

    with patch.object(VRPTimeWindowsSolver, "solve", bad_solve):
        result = benchmarks.bench_vrp_feasibility()

    assert result["capacity_respected_rate"] < 1.0


def test_anomaly_missed_spike():
    """Cover line 206->204: anomaly detector misses a spike."""

    def bad_detect(self, series):
        return [SimpleNamespace(is_anomaly=False) for _ in series]

    with patch.object(AnomalyDetector, "detect_ensemble", bad_detect):
        result = benchmarks.bench_anomaly_detection()

    assert result["spike_recall"] == 0.0


def test_main_all_pass():
    """Cover main() success path."""
    result = benchmarks.main()
    assert result == 0


def test_main_with_error():
    """Cover line 339: benchmark with error."""
    harness = EvaluationHarness()

    def failing_fn():
        raise RuntimeError("test error")

    harness.register(Benchmark("failing", failing_fn, {"metric": 1.0}))

    with patch.object(benchmarks, "build_harness", return_value=harness):
        result = benchmarks.main()

    assert result == 1


def test_main_with_missing_metrics():
    """Cover line 344: benchmark with missing metrics."""
    harness = EvaluationHarness()

    def incomplete_fn():
        return {}

    harness.register(Benchmark("incomplete", incomplete_fn, {"metric": 1.0}))

    with patch.object(benchmarks, "build_harness", return_value=harness):
        result = benchmarks.main()

    assert result == 1


def test_main_entry_point():
    """Cover line 357: __main__ entry point."""
    result = subprocess.run(
        [sys.executable, "-m", "precision_health_os.benchmarks"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
