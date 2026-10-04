"""Concrete benchmarks measuring the platform's real solvers.

These exercise the actual implementations (not mocks) and assert explicit
quality thresholds, so "exceeding benchmarks" is a measured claim.

Run:  python -m precision_health_os.benchmarks
"""

from __future__ import annotations

import itertools
import random
import time

from precision_health_os.evaluation import Benchmark, EvaluationHarness
from precision_health_os.iot import AnomalyDetector
from precision_health_os.optimization import (
    BipartiteMatchingSolver,
    KnapsackSolver,
    Location,
    TSPSolver,
    VRPTimeWindowsSolver,
)

# ---------------------------------------------------------------- helpers


def _tsp_path_len(order: list[int], pts: list[Location]) -> float:
    return sum(
        (
            (pts[order[i]].x - pts[order[i + 1]].x) ** 2
            + (pts[order[i]].y - pts[order[i + 1]].y) ** 2
        )
        ** 0.5
        for i in range(len(order) - 1)
    )


def _nearest_neighbour_order(pts: list[Location]) -> list[int]:
    """Independent NN baseline, computed without the solver under test."""
    unvisited = set(range(len(pts)))
    route = [0]
    unvisited.discard(0)
    while unvisited:
        last = route[-1]
        nxt = min(
            unvisited,
            key=lambda j: (pts[last].x - pts[j].x) ** 2 + (pts[last].y - pts[j].y) ** 2,
        )
        route.append(nxt)
        unvisited.discard(nxt)
    return route


def _exact_tsp_len(pts: list[Location]) -> float:
    """Brute-force optimal tour length from node 0 (small n only)."""
    return min(
        _tsp_path_len([0, *perm], pts) for perm in itertools.permutations(range(1, len(pts)))
    )


# ------------------------------------------------------------ benchmarks


def bench_tsp_quality() -> dict[str, float]:
    """TSP: 2-opt must never be worse than NN, and stay near optimal.

    The optimality gap is only computable by brute force for small n, so it is
    measured on an explicit sample of n=6 instances and divided by the number
    of instances actually solved — not by an assumed count.
    """
    random.seed(1234)  # nosec B311 - benchmark data generation, not security
    solver = TSPSolver()

    # --- monotonicity vs an independent nearest-neighbour baseline ---
    trials = 150
    worse = 0
    for _ in range(trials):
        n = random.randint(6, 10)  # nosec B311 - benchmark data generation, not security
        pts = [
            Location(id=str(i), x=random.uniform(0, 100), y=random.uniform(0, 100))  # nosec B311 - benchmark data generation, not security
            for i in range(n)
        ]
        got = [pts.index(loc) for loc in solver.solve(pts).locations]
        d_got = _tsp_path_len(got, pts)
        d_nn = _tsp_path_len(_nearest_neighbour_order(pts), pts)
        if d_got > d_nn + 1e-9:
            worse += 1

    # --- optimality gap on exact-solvable instances ---
    optimal_trials = 40
    within_10pct = 0
    solved = 0
    for _ in range(optimal_trials):
        pts = [
            Location(id=str(i), x=random.uniform(0, 100), y=random.uniform(0, 100))  # nosec B311 - benchmark data generation, not security
            for i in range(6)
        ]
        got = [pts.index(loc) for loc in solver.solve(pts).locations]
        d_got = _tsp_path_len(got, pts)
        opt = _exact_tsp_len(pts)
        if opt > 0:
            solved += 1
            if (d_got - opt) / opt <= 0.10:
                within_10pct += 1

    return {
        "no_regression_vs_nn": 1.0 - (worse / trials),
        "within_10pct_of_optimal": within_10pct / solved if solved else 0.0,
    }


def bench_knapsack_optimality() -> dict[str, float]:
    """Knapsack DP must match brute-force optimum exactly."""
    random.seed(99)  # nosec B311 - benchmark data generation, not security
    matches = 0
    trials = 60

    for _ in range(trials):
        capacity = random.randint(20, 60)  # nosec B311 - benchmark data generation, not security
        items = [
            {
                "name": f"i{k}",
                "value": random.randint(10, 100),  # nosec B311 - benchmark data generation, not security
                "weight": random.randint(5, 30),  # nosec B311 - benchmark data generation, not security
            }
            for k in range(random.randint(3, 8))  # nosec B311 - benchmark data generation, not security
        ]
        got = sum(it["value"] for it in KnapsackSolver(float(capacity)).solve(items))
        best = max(
            (
                sum(it["value"] for it in combo)
                for r in range(len(items) + 1)
                for combo in itertools.combinations(items, r)
                if sum(it["weight"] for it in combo) <= capacity
            ),
            default=0,
        )
        if got == best:
            matches += 1

    return {"optimality_rate": matches / trials}


def bench_vrp_feasibility() -> dict[str, float]:
    """VRP must serve every node and never breach vehicle capacity."""
    random.seed(7)  # nosec B311 - benchmark data generation, not security
    served_all = 0
    capacity_ok = 0
    trials = 40

    for _ in range(trials):
        depot = Location(id="d", x=0, y=0)
        locs = [
            Location(
                id=f"l{i}",
                x=random.uniform(-20, 20),  # nosec B311 - benchmark data generation, not security
                y=random.uniform(-20, 20),  # nosec B311 - benchmark data generation, not security
                demand=random.randint(5, 40),  # nosec B311 - benchmark data generation, not security
                service_time=5,
                ready_time=0,
                due_time=1000,
            )
            for i in range(random.randint(4, 12))  # nosec B311 - benchmark data generation, not security
        ]
        routes = VRPTimeWindowsSolver(depot, vehicle_capacity=100).solve(locs)
        served = sum(len(r.locations) for r in routes)
        if served == len(locs):
            served_all += 1
        if all(sum(loc.demand for loc in r.locations) <= 100 for r in routes):
            capacity_ok += 1

    return {
        "all_nodes_served_rate": served_all / trials,
        "capacity_respected_rate": capacity_ok / trials,
    }


def bench_bipartite_correctness() -> dict[str, float]:
    """Matching must never pair an ineligible patient with a trial."""
    patients = [
        {"id": "p1", "age": 50, "conditions": ["cancer"]},
        {"id": "p2", "age": 80, "conditions": ["cancer"]},
        {"id": "p3", "age": 40, "conditions": ["diabetes"]},
        {"id": "p4", "age": 30, "conditions": ["cancer"]},
    ]
    trials = [{"id": "t1", "conditions": ["cancer"], "min_age": 18, "max_age": 70}]
    matches = BipartiteMatchingSolver().solve(patients, trials)

    eligible = {"p1", "p4"}  # p2 too old, p3 wrong condition
    paired = {m[0] for m in matches}
    illegal = paired - eligible
    return {
        "no_ineligible_matches": 1.0 if not illegal else 0.0,
        "found_an_eligible_patient": 1.0 if paired else 0.0,
    }


def bench_anomaly_detection() -> dict[str, float]:
    """Detection must catch a real spike and stay quiet on normal variation."""
    detector = AnomalyDetector()

    true_pos = 0
    for spike in (100, 150, 200, 500):
        series = [70, 72, 71, 70, 73, 71, 72, 71, spike]
        if detector.detect_ensemble(series)[-1].is_anomaly:
            true_pos += 1

    true_neg = 0
    for _ in range(20):
        random.seed(_)  # nosec B311 - benchmark data generation, not security
        series = [70 + random.uniform(-3, 3) for _ in range(9)]  # nosec B311 - benchmark data generation, not security
        if not any(r.is_anomaly for r in detector.detect_ensemble(series)):
            true_neg += 1

    return {
        "spike_recall": true_pos / 4,
        "normal_specificity": true_neg / 20,
    }


def bench_solver_latency() -> dict[str, float]:
    """Solvers must stay well inside an interactive latency budget."""
    random.seed(5)  # nosec B311 - benchmark data generation, not security
    pts = [
        Location(id=str(i), x=random.uniform(0, 100), y=random.uniform(0, 100))
        for i in range(25)  # nosec B311 - benchmark data generation, not security
    ]

    start = time.perf_counter()
    TSPSolver().solve(pts)
    tsp_ms = (time.perf_counter() - start) * 1000

    depot = Location(id="d", x=0, y=0)
    locs = [
        Location(
            id=f"l{i}",
            x=random.uniform(-50, 50),  # nosec B311 - benchmark data generation, not security
            y=random.uniform(-50, 50),  # nosec B311 - benchmark data generation, not security
            demand=10,
            service_time=5,
            ready_time=0,
            due_time=1000,
        )
        for i in range(30)
    ]
    start = time.perf_counter()
    VRPTimeWindowsSolver(depot, vehicle_capacity=200).solve(locs)
    vrp_ms = (time.perf_counter() - start) * 1000

    return {"tsp_ms": tsp_ms, "vrp_ms": vrp_ms}


def bench_evaluation_framework() -> dict[str, float]:
    """The framework's own primitives must behave correctly."""
    from precision_health_os.evaluation import EvolutionTracker, Metric

    m_hi = Metric("acc", 0.9, "ratio", True)
    m_lo = Metric("acc", 0.8, "ratio", True)

    tracker = EvolutionTracker(metric="score")
    for gen, score in enumerate([0.40, 0.60, 0.80], start=1):
        tracker.record(gen, score)

    checks = [
        m_hi.better_than(m_lo),
        not m_lo.better_than(m_hi),
        tracker.best_generation() == 3,
        tracker.is_improving(),
        tracker.improvement() > 0,
    ]
    return {"framework_correctness": sum(checks) / len(checks)}


# ---------------------------------------------------------------- harness


def build_harness() -> EvaluationHarness:
    """Build the full benchmark suite with explicit thresholds."""
    harness = EvaluationHarness()
    harness.register(
        Benchmark(
            "tsp_quality",
            bench_tsp_quality,
            {"no_regression_vs_nn": 1.0, "within_10pct_of_optimal": 0.80},
        )
    )
    harness.register(
        Benchmark("knapsack_optimality", bench_knapsack_optimality, {"optimality_rate": 1.0})
    )
    harness.register(
        Benchmark(
            "vrp_feasibility",
            bench_vrp_feasibility,
            {"all_nodes_served_rate": 1.0, "capacity_respected_rate": 1.0},
        )
    )
    harness.register(
        Benchmark(
            "bipartite_correctness",
            bench_bipartite_correctness,
            {"no_ineligible_matches": 1.0, "found_an_eligible_patient": 1.0},
        )
    )
    harness.register(
        Benchmark(
            "anomaly_detection",
            bench_anomaly_detection,
            {"spike_recall": 1.0, "normal_specificity": 0.95},
        )
    )
    harness.register(
        Benchmark(
            "solver_latency",
            bench_solver_latency,
            {"tsp_ms": 2000.0, "vrp_ms": 2000.0},
            lower_is_better={"tsp_ms", "vrp_ms"},
        )
    )
    harness.register(
        Benchmark(
            "evaluation_framework", bench_evaluation_framework, {"framework_correctness": 1.0}
        )
    )
    return harness


def main() -> int:
    """Run the suite and print a report. Returns 0 when all pass."""
    harness = build_harness()
    summary = harness.summary()

    print("=" * 68)
    print("PRECISION HEALTH OS — BENCHMARK SUITE")
    print("=" * 68)
    for r in summary["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{mark}] {r['benchmark_name']}  (score {r['score']:.2f})")
        if r["error"]:
            print(f"        error: {r['error']}")
        for k, v in r["metrics"].items():
            req = r["thresholds"].get(k)
            print(f"        {k}: {v:.4f}  (required {req})")
        if r["missing_metrics"]:
            print(f"        missing: {r['missing_metrics']}")

    print("\n" + "=" * 68)
    print(
        f"total={summary['total']} passed={summary['passed']} "
        f"failed={summary['failed']} pass_rate={summary['pass_rate']:.2%} "
        f"mean_score={summary['mean_score']:.3f}"
    )
    print("=" * 68)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
