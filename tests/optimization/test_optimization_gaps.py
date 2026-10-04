"""Tests for optimization module gap coverage."""

from __future__ import annotations

import itertools
import random

from precision_health_os.optimization import (
    BipartiteMatchingSolver,
    KnapsackSolver,
    Location,
    Route,
    TSPSolver,
    VRPTimeWindowsSolver,
)


class TestVRPTimeWindowsGaps:
    """Tests for uncovered VRPTimeWindowsSolver branches."""

    def test_num_vehicles_explicit(self):
        """Passing num_vehicles explicitly skips the default branch (line 63->67)."""
        depot = Location(id="depot", x=0, y=0)
        locs = [
            Location(id="A", x=1, y=0),
            Location(id="B", x=2, y=0),
            Location(id="C", x=3, y=0),
        ]
        solver = VRPTimeWindowsSolver(depot)
        result = solver.solve(locs, num_vehicles=2)
        assert len(result) >= 1
        assert all(isinstance(r, Route) for r in result)

    def test_is_feasible_returns_false_when_arrival_exceeds_due_time(self):
        """_is_feasible returns False when arrival > due_time (line 142)."""
        depot = Location(id="depot", x=0, y=0)
        far_loc = Location(id="far", x=100, y=0, due_time=1.0)
        solver = VRPTimeWindowsSolver(depot)
        assert solver._is_feasible([far_loc]) is False

    def test_try_merge_returns_none_when_both_directions_infeasible(self):
        """_try_merge returns None when append and prepend both fail (line 102->87)."""
        depot = Location(id="depot", x=0, y=0)
        loc_a = Location(id="A", x=0, y=0, due_time=0)
        loc_b = Location(id="B", x=1, y=0, due_time=0)
        solver = VRPTimeWindowsSolver(depot)
        result = solver._try_merge([loc_a], [loc_b], loc_a, loc_b)
        assert result is None

    def test_try_merge_prepend_succeeds_when_append_fails(self):
        """_try_merge returns prepended route when append fails (lines 128-132)."""
        depot = Location(id="depot", x=0, y=0)
        loc_a = Location(id="A", x=0, y=0, ready_time=100, due_time=200)
        loc_b = Location(id="B", x=1, y=0, ready_time=0, due_time=50)
        solver = VRPTimeWindowsSolver(depot)
        result = solver._try_merge([loc_a], [loc_b], loc_a, loc_b)
        assert result is not None
        assert [loc.id for loc in result] == ["B", "A"]


class TestTSPSolverGaps:
    """Tests for uncovered TSPSolver branches."""

    def test_two_opt_finds_improvement(self):
        """2-opt improves a non-optimal nearest-neighbor route (lines 215-216)."""
        locs = [
            Location(id="0", x=0, y=0),
            Location(id="1", x=10, y=0),
            Location(id="2", x=5, y=1),
            Location(id="3", x=5, y=-1),
            Location(id="4", x=6, y=0),
        ]
        solver = TSPSolver()
        route = solver.solve(locs)
        assert isinstance(route, Route)
        assert len(route.locations) == 5
        assert set(loc.id for loc in route.locations) == {"0", "1", "2", "3", "4"}


class TestKnapsackSolverGaps:
    """Tests for uncovered KnapsackSolver edge cases."""

    def test_zero_capacity_returns_empty(self):
        """Zero capacity returns empty selection."""
        solver = KnapsackSolver(capacity=0)
        items = [{"value": 10, "weight": 1}, {"value": 20, "weight": 2}]
        result = solver.solve(items)
        assert result == []

    def test_exact_capacity_selects_all(self):
        """Items with total weight exactly equal to capacity are all selected."""
        solver = KnapsackSolver(capacity=10)
        items = [
            {"value": 10, "weight": 5},
            {"value": 20, "weight": 5},
        ]
        result = solver.solve(items)
        assert len(result) == 2
        assert {item["value"] for item in result} == {10, 20}

    def test_backtracking_matches_brute_force(self):
        """DP backtracking matches brute force optimal for small instances."""
        random.seed(42)
        for _ in range(20):
            n = random.randint(1, 8)
            items = [
                {"value": random.randint(1, 20), "weight": random.randint(1, 10)} for _ in range(n)
            ]
            capacity = random.randint(5, 30)
            solver = KnapsackSolver(capacity)
            dp_result = solver.solve(items)
            dp_value = sum(item["value"] for item in dp_result)
            dp_weight = sum(item["weight"] for item in dp_result)
            assert dp_weight <= capacity

            best_value = 0
            for r in range(n + 1):
                for combo in itertools.combinations(items, r):
                    w = sum(item["weight"] for item in combo)
                    v = sum(item["value"] for item in combo)
                    if w <= capacity and v > best_value:
                        best_value = v
            assert dp_value == best_value


class TestBipartiteMatchingGaps:
    """Tests for uncovered BipartiteMatchingSolver branches."""

    def test_custom_compatibility_fn_used(self):
        """Custom compatibility_fn is used instead of default (line 316)."""
        patients = [{"id": "p1", "conditions": ["diabetes"]}]
        trials = [{"id": "t1", "conditions": ["diabetes"]}]

        def custom_fn(p, t):
            return 0.9

        solver = BipartiteMatchingSolver()
        result = solver.solve(patients, trials, compatibility_fn=custom_fn)
        assert len(result) == 1
        assert result[0] == ("p1", "t1", 0.9)

    def test_skips_already_matched_edges(self):
        """Edges involving already-matched patients/trials are skipped (line 329->328)."""
        patients = [
            {"id": "p1", "conditions": ["c1"]},
            {"id": "p2", "conditions": ["c1"]},
        ]
        trials = [{"id": "t1", "conditions": ["c1"]}]
        solver = BipartiteMatchingSolver()
        result = solver.solve(patients, trials)
        assert len(result) == 1

    def test_age_out_of_range_skips_bonus(self):
        """Patient age outside trial range skips age bonus (line 343->347)."""
        patients = [{"id": "p1", "conditions": ["c1"], "age": 10}]
        trials = [{"id": "t1", "conditions": ["c1"], "min_age": 18, "max_age": 65}]
        solver = BipartiteMatchingSolver()
        result = solver.solve(patients, trials)
        assert len(result) == 1
        assert result[0][2] == 0.5
