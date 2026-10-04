"""Tests for the benchmark suite itself.

The suite is normally run as ``python -m precision_health_os.benchmarks``,
which coverage does not observe. These tests import and execute the same
functions in-process so the suite is verified as code, not just as a CLI.

They also assert the suite's headline contract: every registered benchmark
passes, which is the platform's "exceeds benchmarks" gate.
"""

from __future__ import annotations

from precision_health_os.benchmarks import (
    bench_anomaly_detection,
    bench_bipartite_correctness,
    bench_evaluation_framework,
    bench_knapsack_optimality,
    bench_solver_latency,
    bench_tsp_quality,
    bench_vrp_feasibility,
    build_harness,
    main,
)


class TestIndividualBenchmarks:
    """Each benchmark returns the metrics the suite expects to threshold."""

    def test_tsp_quality_reports_both_metrics(self) -> None:
        result = bench_tsp_quality()
        assert set(result) == {"no_regression_vs_nn", "within_10pct_of_optimal"}
        assert 0.0 <= result["no_regression_vs_nn"] <= 1.0
        assert 0.0 <= result["within_10pct_of_optimal"] <= 1.0

    def test_tsp_never_regresses_against_nearest_neighbour(self) -> None:
        assert bench_tsp_quality()["no_regression_vs_nn"] == 1.0

    def test_knapsack_matches_brute_force_optimum(self) -> None:
        assert bench_knapsack_optimality()["optimality_rate"] == 1.0

    def test_vrp_serves_all_nodes_within_capacity(self) -> None:
        result = bench_vrp_feasibility()
        assert result["all_nodes_served_rate"] == 1.0
        assert result["capacity_respected_rate"] == 1.0

    def test_bipartite_never_pairs_ineligible_patients(self) -> None:
        result = bench_bipartite_correctness()
        assert result["no_ineligible_matches"] == 1.0
        assert result["found_an_eligible_patient"] == 1.0

    def test_anomaly_detection_recall_and_specificity(self) -> None:
        result = bench_anomaly_detection()
        assert result["spike_recall"] == 1.0
        assert result["normal_specificity"] >= 0.95

    def test_solver_latency_is_interactive(self) -> None:
        result = bench_solver_latency()
        assert result["tsp_ms"] < 2000.0
        assert result["vrp_ms"] < 2000.0

    def test_evaluation_framework_primitives(self) -> None:
        assert bench_evaluation_framework()["framework_correctness"] == 1.0


class TestHarness:
    """The assembled suite must be complete and green."""

    def test_harness_registers_all_benchmarks(self) -> None:
        results = build_harness().run_all()
        names = {r.benchmark_name for r in results}
        assert names == {
            "tsp_quality",
            "knapsack_optimality",
            "vrp_feasibility",
            "bipartite_correctness",
            "anomaly_detection",
            "solver_latency",
            "evaluation_framework",
        }

    def test_every_benchmark_passes(self) -> None:
        """The headline gate: all benchmarks meet their thresholds."""
        summary = build_harness().summary()
        failed = [r["benchmark_name"] for r in summary["results"] if not r["passed"]]
        assert failed == [], f"failing benchmarks: {failed}"

    def test_pass_rate_is_total(self) -> None:
        assert build_harness().summary()["pass_rate"] == 1.0

    def test_mean_score_is_perfect(self) -> None:
        assert build_harness().summary()["mean_score"] == 1.0

    def test_harness_is_deterministic_across_runs(self) -> None:
        """Seeded benchmarks must produce identical verdicts run to run."""
        first = build_harness().summary()
        second = build_harness().summary()
        assert first["passed"] == second["passed"]
        assert first["total"] == second["total"]

    def test_main_exits_zero_when_all_pass(self) -> None:
        assert main() == 0
