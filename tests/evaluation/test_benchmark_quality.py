"""Benchmark-driven quality regression tests.

These encode the thresholds the benchmark suite enforces, as tests. Written
RED against the current implementations, which fail them.

Gap 1: TSPSolver's 2-opt pass had an off-by-one in its sweep bounds, so it
       missed legal improving moves and only reached within 10% of the exact
       optimum on 60% of n=6 instances.
Gap 2: AnomalyDetector's ensemble produced ~10% false positives on stationary
       noise (2 of 20 random series flagged), above the 5% budget.
"""

from __future__ import annotations

import itertools
import random

from precision_health_os.iot import AnomalyDetector
from precision_health_os.optimization import Location, TSPSolver


def _path_len(order: list[int], pts: list[Location]) -> float:
    return sum(
        (
            (pts[order[i]].x - pts[order[i + 1]].x) ** 2
            + (pts[order[i]].y - pts[order[i + 1]].y) ** 2
        )
        ** 0.5
        for i in range(len(order) - 1)
    )


def _exact_opt(pts: list[Location]) -> float:
    return min(_path_len([0, *p], pts) for p in itertools.permutations(range(1, len(pts))))


class TestTSPOptimalityGap:
    """2-opt must get close to optimal, not just beat nearest-neighbour."""

    def test_within_10pct_of_optimal_on_most_instances(self) -> None:
        """At least 80% of n=6 instances must land within 10% of optimal."""
        random.seed(1234)
        solver = TSPSolver()
        within = 0
        trials = 100

        for _ in range(trials):
            pts = [
                Location(id=str(i), x=random.uniform(0, 100), y=random.uniform(0, 100))
                for i in range(6)
            ]
            got = [pts.index(loc) for loc in solver.solve(pts).locations]
            d_got = _path_len(got, pts)
            opt = _exact_opt(pts)
            if opt > 0 and (d_got - opt) / opt <= 0.10:
                within += 1

        rate = within / trials
        assert rate >= 0.80, f"only {rate:.0%} within 10% of optimal (need 80%)"

    def test_two_opt_improves_a_known_crossing_route(self) -> None:
        """A route with a visible crossing must be untangled by 2-opt.

        Points on a line in shuffled order: the optimal tour follows the line,
        so any remaining crossing means 2-opt stopped short.
        """
        pts = [
            Location(id="a", x=0, y=0),
            Location(id="b", x=3, y=0),
            Location(id="c", x=1, y=0),
            Location(id="d", x=2, y=0),
            Location(id="e", x=4, y=0),
        ]
        got = [pts.index(loc) for loc in TSPSolver().solve(pts).locations]
        d_got = _path_len(got, pts)
        opt = _exact_opt(pts)
        assert d_got <= opt * 1.01, f"2-opt left a crossing: {d_got:.3f} vs optimal {opt:.3f}"

    def test_never_worse_than_nearest_neighbour(self) -> None:
        """2-opt must be monotone: never return a worse tour than its NN seed."""
        random.seed(555)
        solver = TSPSolver()
        worse = 0

        for _ in range(100):
            n = random.randint(6, 10)
            pts = [
                Location(id=str(i), x=random.uniform(0, 100), y=random.uniform(0, 100))
                for i in range(n)
            ]
            got = [pts.index(loc) for loc in solver.solve(pts).locations]

            # independent NN baseline
            unv = set(range(n))
            nn = [0]
            unv.discard(0)
            while unv:
                last = nn[-1]
                nxt = min(
                    unv,
                    key=lambda j: (pts[last].x - pts[j].x) ** 2 + (pts[last].y - pts[j].y) ** 2,
                )
                nn.append(nxt)
                unv.discard(nxt)

            if _path_len(got, pts) > _path_len(nn, pts) + 1e-9:
                worse += 1

        assert worse == 0, f"2-opt returned a worse tour than NN on {worse}/100 instances"


class TestAnomalySpecificityGap:
    """The ensemble must not cry wolf on stationary noise."""

    def test_false_positive_rate_under_5_percent(self) -> None:
        """At most 5% of stationary random series may be flagged."""
        detector = AnomalyDetector()
        flagged = 0
        trials = 40

        for seed in range(trials):
            random.seed(seed)
            series = [70 + random.uniform(-3, 3) for _ in range(9)]
            if any(r.is_anomaly for r in detector.detect_ensemble(series)):
                flagged += 1

        rate = flagged / trials
        assert rate <= 0.05, f"false-positive rate {rate:.0%} exceeds 5% budget"

    def test_still_catches_every_real_spike(self) -> None:
        """Tightening specificity must not cost recall."""
        detector = AnomalyDetector()
        for spike in (100, 150, 200, 500):
            series = [70, 72, 71, 70, 73, 71, 72, 71, spike]
            assert detector.detect_ensemble(series)[-1].is_anomaly, f"missed spike {spike}"

    def test_constant_signal_is_never_flagged(self) -> None:
        detector = AnomalyDetector()
        assert not any(r.is_anomaly for r in detector.detect_ensemble([70.0] * 10))

    def test_small_physiological_variation_not_flagged(self) -> None:
        """Normal HR wobble (60-90 bpm) must not alert."""
        detector = AnomalyDetector()
        series = [68, 72, 75, 71, 69, 73, 76, 70, 72, 74, 71, 69]
        assert not any(r.is_anomaly for r in detector.detect_ensemble(series))
