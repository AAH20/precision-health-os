"""Tests for uncovered lines in optimization module."""

from precision_health_os.optimization import (
    Location,
    TSPSolver,
    VRPTimeWindowsSolver,
)


def test_vrp_merge_success_covers_line_102():
    """Two close locations with compatible time windows should merge into one route."""
    depot = Location(id="depot", x=0.0, y=0.0)
    # Two locations very close to each other and to depot
    loc_a = Location(id="A", x=1.0, y=0.0, demand=1.0, ready_time=0.0, due_time=100.0)
    loc_b = Location(id="B", x=2.0, y=0.0, demand=1.0, ready_time=0.0, due_time=100.0)

    solver = VRPTimeWindowsSolver(depot=depot, vehicle_capacity=100.0)
    routes = solver.solve([loc_a, loc_b])

    # Should merge into a single route (line 102: if merged)
    assert len(routes) == 1
    assert len(routes[0].locations) == 2


def test_tsp_route_distance_covers_lines_244_250():
    """_route_distance should sum consecutive edge distances."""
    loc0 = Location(id="0", x=0.0, y=0.0)
    loc1 = Location(id="1", x=3.0, y=0.0)
    loc2 = Location(id="2", x=3.0, y=4.0)

    solver = TSPSolver()
    route = [0, 1, 2]
    dist = solver._route_distance(route, [loc0, loc1, loc2])

    # 3.0 (0->1) + 4.0 (1->2) = 7.0
    assert dist == 7.0


def test_tsp_route_distance_single_node():
    """_route_distance with a single node should return 0."""
    loc = Location(id="0", x=5.0, y=5.0)
    solver = TSPSolver()
    assert solver._route_distance([0], [loc]) == 0.0


def test_vrp_merge_failure_covers_line_102_branch():
    """Two locations too far apart with tight windows cannot merge (line 102->87)."""
    depot = Location(id="depot", x=0.0, y=0.0)
    # Each location is reachable from depot, but visiting both in sequence
    # exceeds the due_time of the second location.
    loc_a = Location(id="A", x=10.0, y=0.0, demand=1.0, ready_time=0.0, due_time=10.0)
    loc_b = Location(id="B", x=20.0, y=0.0, demand=1.0, ready_time=0.0, due_time=10.0)

    solver = VRPTimeWindowsSolver(depot=depot, vehicle_capacity=100.0)
    routes = solver.solve([loc_a, loc_b])

    # Merge fails, so each location stays in its own route
    assert len(routes) == 2
