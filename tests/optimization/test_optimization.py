"""Tests for optimization module."""

from precision_health_os.optimization import (
    BipartiteMatchingSolver,
    KnapsackSolver,
    Location,
    TSPSolver,
    VRPTimeWindowsSolver,
)


class TestVRPTimeWindowsSolver:
    """Tests for VRP with Time Windows solver."""

    def test_empty_locations(self) -> None:
        depot = Location(id="depot", x=0, y=0)
        solver = VRPTimeWindowsSolver(depot)
        result = solver.solve([])
        assert result == []

    def test_single_location(self) -> None:
        depot = Location(id="depot", x=0, y=0)
        loc = Location(id="loc1", x=3, y=4, demand=1, service_time=10)
        solver = VRPTimeWindowsSolver(depot)
        result = solver.solve([loc])
        assert len(result) == 1
        assert len(result[0].locations) == 1

    def test_multiple_locations(self) -> None:
        depot = Location(id="depot", x=0, y=0)
        locations = [
            Location(id="loc1", x=1, y=0, demand=1, service_time=5),
            Location(id="loc2", x=0, y=1, demand=1, service_time=5),
            Location(id="loc3", x=-1, y=0, demand=1, service_time=5),
        ]
        solver = VRPTimeWindowsSolver(depot)
        result = solver.solve(locations)
        assert len(result) >= 1
        total_locs = sum(len(r.locations) for r in result)
        assert total_locs == 3

    def test_capacity_constraint(self) -> None:
        depot = Location(id="depot", x=0, y=0)
        locations = [Location(id=f"loc{i}", x=i, y=0, demand=60, service_time=5) for i in range(5)]
        solver = VRPTimeWindowsSolver(depot, vehicle_capacity=100)
        result = solver.solve(locations)
        for route in result:
            total_demand = sum(loc.demand for loc in route.locations)
            assert total_demand <= 100

    def test_time_window_feasibility(self) -> None:
        depot = Location(id="depot", x=0, y=0)
        locations = [
            Location(id="loc1", x=1, y=0, demand=1, service_time=5, ready_time=0, due_time=100),
            Location(id="loc2", x=2, y=0, demand=1, service_time=5, ready_time=0, due_time=100),
        ]
        solver = VRPTimeWindowsSolver(depot)
        result = solver.solve(locations)
        for route in result:
            assert route.feasible


class TestTSPSolver:
    """Tests for TSP solver."""

    def test_empty_route(self) -> None:
        solver = TSPSolver()
        result = solver.solve([])
        assert result.locations == []

    def test_single_location(self) -> None:
        solver = TSPSolver()
        loc = Location(id="loc1", x=5, y=5)
        result = solver.solve([loc])
        assert len(result.locations) == 1

    def test_two_locations(self) -> None:
        solver = TSPSolver()
        locations = [
            Location(id="loc1", x=0, y=0),
            Location(id="loc2", x=3, y=4),
        ]
        result = solver.solve(locations)
        assert len(result.locations) == 2
        assert result.total_distance > 0

    def test_route_optimization(self) -> None:
        solver = TSPSolver()
        locations = [
            Location(id="loc1", x=0, y=0),
            Location(id="loc2", x=1, y=0),
            Location(id="loc3", x=2, y=0),
            Location(id="loc4", x=3, y=0),
        ]
        result = solver.solve(locations)
        assert len(result.locations) == 4
        # Optimal route should be ~6 units (0->1->2->3)
        assert result.total_distance <= 6.5


class TestKnapsackSolver:
    """Tests for Knapsack solver."""

    def test_empty_items(self) -> None:
        solver = KnapsackSolver(capacity=100)
        result = solver.solve([])
        assert result == []

    def test_single_item_fits(self) -> None:
        solver = KnapsackSolver(capacity=100)
        items = [{"name": "item1", "value": 50, "weight": 30}]
        result = solver.solve(items)
        assert len(result) == 1

    def test_single_item_too_heavy(self) -> None:
        solver = KnapsackSolver(capacity=10)
        items = [{"name": "item1", "value": 50, "weight": 30}]
        result = solver.solve(items)
        assert len(result) == 0

    def test_optimal_selection(self) -> None:
        solver = KnapsackSolver(capacity=50)
        items = [
            {"name": "item1", "value": 60, "weight": 10},
            {"name": "item2", "value": 100, "weight": 20},
            {"name": "item3", "value": 120, "weight": 30},
        ]
        result = solver.solve(items)
        total_weight = sum(item["weight"] for item in result)
        total_value = sum(item["value"] for item in result)
        assert total_weight <= 50
        # Optimal: item2 + item3 = 220 value, 50 weight
        assert total_value == 220


class TestBipartiteMatchingSolver:
    """Tests for Bipartite Matching solver."""

    def test_empty_inputs(self) -> None:
        solver = BipartiteMatchingSolver()
        assert solver.solve([], []) == []
        assert solver.solve([{"id": "p1"}], []) == []
        assert solver.solve([], [{"id": "t1"}]) == []

    def test_simple_match(self) -> None:
        solver = BipartiteMatchingSolver()
        patients = [
            {"id": "p1", "conditions": ["cancer"], "age": 50},
            {"id": "p2", "conditions": ["diabetes"], "age": 60},
        ]
        trials = [
            {"id": "t1", "conditions": ["cancer"], "min_age": 40, "max_age": 70},
            {"id": "t2", "conditions": ["diabetes"], "min_age": 50, "max_age": 70},
        ]
        result = solver.solve(patients, trials)
        assert len(result) == 2

    def test_partial_match(self) -> None:
        solver = BipartiteMatchingSolver()
        patients = [
            {"id": "p1", "conditions": ["cancer"], "age": 50},
        ]
        trials = [
            {"id": "t1", "conditions": ["diabetes"], "min_age": 40, "max_age": 70},
        ]
        result = solver.solve(patients, trials)
        assert len(result) == 0
