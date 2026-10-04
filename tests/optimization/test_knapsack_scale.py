"""Tests for KnapsackSolver backtracking correctness and scalability."""

import itertools
import random
import time
import tracemalloc

from precision_health_os.optimization import KnapsackSolver


def brute_force_knapsack(items, capacity, scale=100):
    """Brute force 0/1 knapsack with same weight scaling as KnapsackSolver."""
    w = int(capacity * scale)
    weights = [int(item["weight"] * scale) for item in items]
    best_value = 0
    best_subset = []
    for r in range(len(items) + 1):
        for subset in itertools.combinations(range(len(items)), r):
            total_weight = sum(weights[i] for i in subset)
            if total_weight <= w:
                total_value = sum(items[i]["value"] for i in subset)
                if total_value > best_value:
                    best_value = total_value
                    best_subset = [items[i] for i in subset]
    return best_subset


class TestKnapsackBacktracking:
    """Tests for KnapsackSolver backtracking correctness."""

    def test_exact_capacity_boundary(self):
        """Items that exactly fill capacity should be selected."""
        solver = KnapsackSolver(capacity=50)
        items = [
            {"name": "a", "value": 60, "weight": 10},
            {"name": "b", "value": 100, "weight": 20},
            {"name": "c", "value": 120, "weight": 30},
        ]
        result = solver.solve(items)
        total_weight = sum(item["weight"] for item in result)
        total_value = sum(item["value"] for item in result)
        assert total_weight == 50
        assert total_value == 220

    def test_zero_value_items(self):
        """Items with zero value should not affect optimal solution."""
        solver = KnapsackSolver(capacity=50)
        items = [
            {"name": "a", "value": 0, "weight": 10},
            {"name": "b", "value": 100, "weight": 20},
            {"name": "c", "value": 120, "weight": 30},
        ]
        result = solver.solve(items)
        total_value = sum(item["value"] for item in result)
        assert total_value == 220

    def test_duplicate_weights(self):
        """Items with same weight but different values."""
        solver = KnapsackSolver(capacity=50)
        items = [
            {"name": "a", "value": 50, "weight": 25},
            {"name": "b", "value": 100, "weight": 25},
            {"name": "c", "value": 60, "weight": 25},
        ]
        result = solver.solve(items)
        total_weight = sum(item["weight"] for item in result)
        total_value = sum(item["value"] for item in result)
        assert total_weight <= 50
        assert total_value == 160  # b + c

    def test_backtracking_vs_bruteforce_random(self):
        """Verify backtracking correctness against brute force on 100 random instances."""
        rng = random.Random(42)
        for _ in range(100):
            n = rng.randint(1, 10)
            capacity = rng.randint(1, 50)
            items = [
                {"name": f"item{i}", "value": rng.randint(1, 100), "weight": rng.randint(1, 20)}
                for i in range(n)
            ]
            solver = KnapsackSolver(capacity=capacity)
            result = solver.solve(items)
            expected = brute_force_knapsack(items, capacity)
            result_value = sum(item["value"] for item in result)
            expected_value = sum(item["value"] for item in expected)
            assert result_value == expected_value

    def test_fractional_weights_and_capacity(self):
        """Solver should handle fractional weights via scaling."""
        solver = KnapsackSolver(capacity=10.5)
        items = [
            {"name": "a", "value": 10, "weight": 5.5},
            {"name": "b", "value": 20, "weight": 5.0},
        ]
        result = solver.solve(items)
        total_weight = sum(item["weight"] for item in result)
        total_value = sum(item["value"] for item in result)
        assert total_weight <= 10.5
        assert total_value == 30

    def test_all_items_too_heavy(self):
        """Should return empty list when no items fit."""
        solver = KnapsackSolver(capacity=5)
        items = [
            {"name": "a", "value": 10, "weight": 10},
            {"name": "b", "value": 20, "weight": 20},
        ]
        result = solver.solve(items)
        assert result == []


class TestKnapsackScalability:
    """Tests for KnapsackSolver scalability."""

    def test_large_capacity_completes_in_time(self):
        """Large capacity case should complete within sane time budget."""
        rng = random.Random(123)
        items = [
            {"name": f"item{i}", "value": rng.randint(1, 1000), "weight": rng.randint(1, 100)}
            for i in range(20)
        ]
        solver = KnapsackSolver(capacity=1000)
        start = time.time()
        result = solver.solve(items)
        elapsed = time.time() - start
        assert elapsed < 30
        total_weight = sum(item["weight"] for item in result)
        assert total_weight <= 1000

    def test_large_capacity_memory_usage(self):
        """Large capacity case should not consume excessive memory."""
        rng = random.Random(456)
        items = [
            {"name": f"item{i}", "value": rng.randint(1, 1000), "weight": rng.randint(1, 100)}
            for i in range(20)
        ]
        solver = KnapsackSolver(capacity=1000)
        tracemalloc.start()
        solver.solve(items)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 200 * 1024 * 1024
