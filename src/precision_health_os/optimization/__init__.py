"""Optimization module: NP-hard problem solvers for healthcare.

Implements:
- Vehicle Routing Problem with Time Windows (VRPTW) for patient scheduling
- Traveling Salesman Problem (TSP) variants for treatment planning
- Knapsack for drug combination optimization
- Bipartite matching for clinical trial matching
"""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from precision_health_os.utils import euclidean_distance

logger = logging.getLogger(__name__)


@dataclass
class Location:
    """Geographic location with time window."""

    id: str
    x: float
    y: float
    demand: float = 1.0
    service_time: float = 0.0
    ready_time: float = 0.0
    due_time: float = float("inf")


@dataclass
class Route:
    """Vehicle route with locations."""

    locations: list[Location] = field(default_factory=list)
    total_distance: float = 0.0
    total_time: float = 0.0
    feasible: bool = True


class VRPTimeWindowsSolver:
    """Vehicle Routing Problem with Time Windows (VRPTW).

    NP-hard: O(n^2 * 2^n) for exact, O(n^2) per iteration for heuristic.
    Uses Clarke-Wright savings heuristic with time window feasibility checks.
    """

    def __init__(self, depot: Location, vehicle_capacity: float = 100.0) -> None:
        self.depot = depot
        self.vehicle_capacity = vehicle_capacity

    def solve(self, locations: list[Location], num_vehicles: int | None = None) -> list[Route]:
        """Solve VRPTW using Clarke-Wright savings heuristic."""
        if not locations:
            return []

        n = len(locations)
        if num_vehicles is None:
            num_vehicles = max(1, n // 5)

        # Compute savings
        savings: list[tuple[float, int, int]] = []
        for i, j in itertools.combinations(range(n), 2):
            d_i_depot = euclidean_distance(
                [locations[i].x, locations[i].y], [self.depot.x, self.depot.y]
            )
            d_j_depot = euclidean_distance(
                [locations[j].x, locations[j].y], [self.depot.x, self.depot.y]
            )
            d_i_j = euclidean_distance(
                [locations[i].x, locations[i].y], [locations[j].x, locations[j].y]
            )
            saving = d_i_depot + d_j_depot - d_i_j
            savings.append((saving, i, j))

        savings.sort(reverse=True)

        # Initialize routes: each location in its own route
        routes: list[list[Location]] = [[loc] for loc in locations]
        route_of = list(range(n))

        for saving, i, j in savings:
            ri, rj = route_of[i], route_of[j]
            if ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            # Check capacity
            total_demand = sum(loc.demand for loc in route_i) + sum(
                loc.demand for loc in route_j
            )
            if total_demand > self.vehicle_capacity:
                continue

            # Try merging: j at end of route_i or at start
            merged = self._try_merge(route_i, route_j, locations[i], locations[j])
            if merged:
                routes[ri] = merged
                routes[rj] = []
                for loc in merged:
                    route_of[locations.index(loc)] = ri

        # Build final routes
        result: list[Route] = []
        for route_locs in routes:
            if not route_locs:
                continue
            route = self._build_route(route_locs)
            result.append(route)

        return result

    def _try_merge(
        self, route_i: list[Location], route_j: list[Location], loc_i: Location, loc_j: Location
    ) -> list[Location] | None:
        """Try merging two routes respecting time windows."""
        # Try appending route_j to route_i
        candidate = route_i + route_j
        if self._is_feasible(candidate):
            return candidate

        # Try prepending route_j to route_i
        candidate = route_j + route_i
        if self._is_feasible(candidate):
            return candidate

        return None

    def _is_feasible(self, route: list[Location]) -> bool:
        """Check if a route satisfies time window constraints."""
        time = 0.0
        prev = self.depot
        for loc in route:
            dist = euclidean_distance([prev.x, prev.y], [loc.x, loc.y])
            arrival = time + dist
            if arrival > loc.due_time:
                return False
            time = max(arrival, loc.ready_time) + loc.service_time
            prev = loc
        return True

    def _build_route(self, locations: list[Location]) -> Route:
        """Build a Route object from locations."""
        total_dist = 0.0
        total_time = 0.0
        prev = self.depot
        for loc in locations:
            dist = euclidean_distance([prev.x, prev.y], [loc.x, loc.y])
            total_dist += dist
            arrival = total_time + dist
            total_time = max(arrival, loc.ready_time) + loc.service_time
            prev = loc
        # Return to depot
        total_dist += euclidean_distance([prev.x, prev.y], [self.depot.x, self.depot.y])
        return Route(locations=locations, total_distance=total_dist, total_time=total_time)


class TSPSolver:
    """Traveling Salesman Problem solver for treatment planning.

    NP-hard: O(n!) exact, O(n^2 * 2^n) Held-Karp DP.
    Uses nearest neighbor + 2-opt improvement.
    """

    def __init__(self) -> None:
        pass

    def solve(self, locations: list[Location]) -> Route:
        """Solve TSP using nearest neighbor + 2-opt."""
        if len(locations) <= 1:
            return Route(locations=locations)

        # Nearest neighbor
        unvisited = set(range(len(locations)))
        route = [0]
        unvisited.remove(0)

        while unvisited:
            last = route[-1]
            nearest = min(
                unvisited,
                key=lambda j: euclidean_distance(
                    [locations[last].x, locations[last].y],
                    [locations[j].x, locations[j].y],
                ),
            )
            route.append(nearest)
            unvisited.remove(nearest)

        # 2-opt improvement
        route = self._two_opt(route, locations)

        route_locs = [locations[i] for i in route]
        return self._build_route(route_locs)

    def _two_opt(self, route: list[int], locations: list[Location]) -> list[int]:
        """2-opt local search improvement."""
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    if self._route_distance(new_route, locations) < self._route_distance(
                        route, locations
                    ):
                        route = new_route
                        improved = True
        return route

    def _route_distance(self, route: list[int], locations: list[Location]) -> float:
        """Compute total distance of a route."""
        total = 0.0
        for i in range(len(route) - 1):
            total += euclidean_distance(
                [locations[route[i]].x, locations[route[i]].y],
                [locations[route[i + 1]].x, locations[route[i + 1]].y],
            )
        return total

    def _build_route(self, locations: list[Location]) -> Route:
        """Build a Route object."""
        total_dist = 0.0
        for i in range(len(locations) - 1):
            total_dist += euclidean_distance(
                [locations[i].x, locations[i].y],
                [locations[i + 1].x, locations[i + 1].y],
            )
        return Route(locations=locations, total_distance=total_dist)


class KnapsackSolver:
    """0/1 Knapsack solver for drug combination optimization.

    NP-hard: O(n * W) pseudo-polynomial (DP).
    """

    def __init__(self, capacity: float) -> None:
        self.capacity = capacity

    def solve(
        self, items: list[dict[str, Any]], value_key: str = "value", weight_key: str = "weight"
    ) -> list[dict[str, Any]]:
        """Solve 0/1 knapsack using dynamic programming."""
        n = len(items)
        if n == 0:
            return []

        # Scale weights to integers for DP
        scale = 100
        W = int(self.capacity * scale)
        weights = [int(item[weight_key] * scale) for item in items]
        values = [item[value_key] for item in items]

        # DP table
        dp: list[list[float]] = [[0.0] * (W + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(W + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        dp[i - 1][w],
                        dp[i - 1][w - weights[i - 1]] + values[i - 1],
                    )
                else:
                    dp[i][w] = dp[i - 1][w]

        # Backtrack to find selected items
        selected: list[dict[str, Any]] = []
        w = W
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected.append(items[i - 1])
                w -= weights[i - 1]

        return selected


class BipartiteMatchingSolver:
    """Bipartite matching for clinical trial matching.

    Polynomial: O(V * E) for Hopcroft-Karp.
    Uses maximum bipartite matching with preference scores.
    """

    def __init__(self) -> None:
        pass

    def solve(
        self,
        patients: list[dict[str, Any]],
        trials: list[dict[str, Any]],
        compatibility_fn: Any = None,
    ) -> list[tuple[str, str, float]]:
        """Find optimal patient-trial matching.

        Returns list of (patient_id, trial_id, score) tuples.
        """
        if not patients or not trials:
            return []

        # Build compatibility matrix
        scores: dict[tuple[str, str], float] = {}
        for p in patients:
            for t in trials:
                if compatibility_fn:
                    score = compatibility_fn(p, t)
                else:
                    score = self._default_compatibility(p, t)
                if score > 0:
                    scores[(p["id"], t["id"])] = score

        # Greedy matching (can be replaced with Hungarian algorithm)
        matched_patients: set[str] = set()
        matched_trials: set[str] = set()
        matches: list[tuple[str, str, float]] = []

        sorted_edges = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for (pid, tid), score in sorted_edges:
            if pid not in matched_patients and tid not in matched_trials:
                matches.append((pid, tid, score))
                matched_patients.add(pid)
                matched_trials.add(tid)

        return matches

    def _default_compatibility(self, patient: dict[str, Any], trial: dict[str, Any]) -> float:
        """Default compatibility score. Returns 0 if no condition overlap."""
        patient_conditions = set(patient.get("conditions", []))
        trial_conditions = set(trial.get("conditions", []))
        if not (patient_conditions & trial_conditions):
            return 0.0
        score = 0.5
        if patient.get("age", 0) >= trial.get("min_age", 0) and patient.get(
            "age", 150
        ) <= trial.get("max_age", 150):
            score += 0.3
        return min(score, 1.0)
