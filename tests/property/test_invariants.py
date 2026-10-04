"""Property-based invariant tests using hypothesis.

These tests assert relationships that must hold for ALL inputs,
not specific values. They cover the solver modules, crypto, and utils.
"""

from __future__ import annotations

import itertools
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from precision_health_os.optimization import (
    KnapsackSolver,
    Location,
    TSPSolver,
    VRPTimeWindowsSolver,
)
from precision_health_os.security import AuditTrail, EncryptionService
from precision_health_os.utils import normalize, z_score

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

text_strategy = st.text(min_size=1, max_size=200)
positive_float_strategy = st.floats(
    min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
)
non_negative_float_strategy = st.floats(
    min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
)
float_strategy = st.floats(
    min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)
int_strategy = st.integers(min_value=1, max_value=50)


def make_locations(n: int, x_strategy=None, y_strategy=None) -> list[Location]:
    """Build n distinct Location objects."""
    x_strategy = x_strategy or st.floats(
        min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
    )
    y_strategy = y_strategy or st.floats(
        min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
    )
    return [
        Location(id=f"loc_{i}", x=float(x), y=float(y))
        for i, (x, y) in enumerate(zip(x_strategy, y_strategy, strict=False))
    ]


locations_strategy = st.lists(
    st.builds(
        Location,
        id=st.text(min_size=1, max_size=10),
        x=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        demand=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    ),
    min_size=2,
    max_size=12,
    unique_by=lambda loc: loc.id,
)


# ---------------------------------------------------------------------------
# 1. EncryptionService
# ---------------------------------------------------------------------------


class TestEncryptionService:
    """Invariants for AES-256-GCM encryption."""

    @given(plaintext=text_strategy)
    @settings(max_examples=50, deadline=None)
    def test_decrypt_encrypt_roundtrip(self, plaintext: str) -> None:
        """decrypt(encrypt(x)) == x for any non-empty text."""
        key = b"\x00" * 32
        svc = EncryptionService(key)
        ciphertext = svc.encrypt(plaintext)
        assert svc.decrypt(ciphertext) == plaintext

    @given(plaintext=text_strategy)
    @settings(max_examples=50, deadline=None)
    def test_nonce_uniqueness(self, plaintext: str) -> None:
        """Two encryptions of the same text must differ (nonce uniqueness)."""
        key = b"\x00" * 32
        svc = EncryptionService(key)
        ct1 = svc.encrypt(plaintext)
        ct2 = svc.encrypt(plaintext)
        assert ct1 != ct2


# ---------------------------------------------------------------------------
# 2. AuditTrail
# ---------------------------------------------------------------------------


class TestAuditTrail:
    """Invariants for the hash-chain audit trail."""

    @given(
        events=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10),
                st.text(min_size=1, max_size=10),
                st.text(min_size=1, max_size=10),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_verify_chain_true_for_valid_sequence(self, events: list[tuple[str, str, str]]) -> None:
        """verify_chain() is True after logging any sequence of events."""
        trail = AuditTrail()
        for user_id, action, resource_id in events:
            trail.log(user_id, action, "patient", resource_id)
        assert trail.verify_chain() is True

    @given(
        events=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10),
                st.text(min_size=1, max_size=10),
                st.text(min_size=1, max_size=10),
            ),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_verify_chain_false_after_mutation(self, events: list[tuple[str, str, str]]) -> None:
        """Mutating any single event breaks the chain."""
        trail = AuditTrail()
        for user_id, action, resource_id in events:
            trail.log(user_id, action, "patient", resource_id)
        assert trail.verify_chain() is True

        # Mutate the first event's action
        trail.get_events()[0].action = "tampered"
        assert trail.verify_chain() is False


# ---------------------------------------------------------------------------
# 3. KnapsackSolver
# ---------------------------------------------------------------------------


class TestKnapsackSolver:
    """Invariants for the 0/1 knapsack DP solver."""

    @given(
        items=st.lists(
            st.fixed_dictionaries(
                {
                    "value": st.floats(
                        min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
                    ),
                    "weight": st.floats(
                        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
                    ),
                }
            ),
            min_size=1,
            max_size=10,
        ),
        capacity=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_weight_never_exceeds_capacity(self, items: list[dict], capacity: float) -> None:
        """Total weight of selected items never exceeds capacity."""
        solver = KnapsackSolver(capacity)
        selected = solver.solve(items)
        total_weight = sum(item["weight"] for item in selected)
        # Tolerance for integer scaling (scale=100, max error per item = 0.01)
        assert total_weight <= capacity + len(items) * 0.01 + 1e-9

    @given(
        items=st.lists(
            st.fixed_dictionaries(
                {
                    "value": st.floats(
                        min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
                    ),
                    "weight": st.floats(
                        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
                    ),
                }
            ),
            min_size=1,
            max_size=10,
        ),
        capacity=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_value_never_exceeds_brute_force_optimum(
        self, items: list[dict], capacity: float
    ) -> None:
        """Selected value never exceeds the brute-force optimum."""
        solver = KnapsackSolver(capacity)
        selected = solver.solve(items)
        selected_value = sum(item["value"] for item in selected)

        # Brute-force optimum using same scaled weights
        scale = 100
        w = int(capacity * scale)
        weights = [int(item["weight"] * scale) for item in items]
        values = [item["value"] for item in items]

        best = 0.0
        for r in range(len(items) + 1):
            for combo in itertools.combinations(range(len(items)), r):
                tw = sum(weights[i] for i in combo)
                if tw <= w:
                    tv = sum(values[i] for i in combo)
                    if tv > best:
                        best = tv

        assert selected_value <= best + 1e-9


# ---------------------------------------------------------------------------
# 4. TSPSolver
# ---------------------------------------------------------------------------


class TestTSPSolver:
    """Invariants for the TSP nearest-neighbor + 2-opt solver."""

    # Filter out degenerate cases where locations are so close that
    # floating-point underflow makes the distance 0.0. The "> 0"
    # invariant only applies when points are meaningfully separated.
    non_degenerate_locations = locations_strategy.filter(
        lambda locs: (
            min(math.hypot(a.x - b.x, a.y - b.y) for a, b in itertools.combinations(locs, 2))
            > 1e-10
        )
    )

    @given(locations=non_degenerate_locations)
    @settings(max_examples=50, deadline=None)
    def test_tour_visits_every_location_exactly_once(self, locations: list[Location]) -> None:
        """The returned tour visits every location exactly once."""
        solver = TSPSolver()
        route = solver.solve(locations)
        assert len(route.locations) == len(locations)
        input_ids = sorted(loc.id for loc in locations)
        route_ids = sorted(loc.id for loc in route.locations)
        assert route_ids == input_ids

    @given(locations=non_degenerate_locations)
    @settings(max_examples=50, deadline=None)
    def test_total_distance_finite_and_positive(self, locations: list[Location]) -> None:
        """total_distance is finite and > 0 for n >= 2 (non-degenerate)."""
        solver = TSPSolver()
        route = solver.solve(locations)
        assert math.isfinite(route.total_distance)
        assert route.total_distance > 0.0


# ---------------------------------------------------------------------------
# 5. VRPTimeWindowsSolver
# ---------------------------------------------------------------------------


class TestVRPTimeWindowsSolver:
    """Invariants for the Clarke-Wright VRPTW solver."""

    @given(
        locations=st.lists(
            st.builds(
                Location,
                id=st.text(min_size=1, max_size=10),
                x=st.floats(
                    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
                ),
                y=st.floats(
                    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
                ),
                demand=st.floats(
                    min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
                ),
            ),
            min_size=2,
            max_size=10,
            unique_by=lambda loc: loc.id,
        ),
        capacity=st.floats(min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_every_location_in_exactly_one_route(
        self, locations: list[Location], capacity: float
    ) -> None:
        """Every location appears in exactly one route."""
        depot = Location(id="depot", x=0.0, y=0.0)
        solver = VRPTimeWindowsSolver(depot, vehicle_capacity=capacity)
        routes = solver.solve(locations)

        all_route_locs = [loc for route in routes for loc in route.locations]
        input_ids = sorted(loc.id for loc in locations)
        route_ids = sorted(loc.id for loc in all_route_locs)
        assert route_ids == input_ids

    @given(
        locations=st.lists(
            st.builds(
                Location,
                id=st.text(min_size=1, max_size=10),
                x=st.floats(
                    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
                ),
                y=st.floats(
                    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
                ),
                demand=st.floats(
                    min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
                ),
            ),
            min_size=2,
            max_size=10,
            unique_by=lambda loc: loc.id,
        ),
        capacity=st.floats(min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_no_route_exceeds_capacity(self, locations: list[Location], capacity: float) -> None:
        """No route's total demand exceeds vehicle capacity."""
        depot = Location(id="depot", x=0.0, y=0.0)
        solver = VRPTimeWindowsSolver(depot, vehicle_capacity=capacity)
        routes = solver.solve(locations)

        for route in routes:
            total_demand = sum(loc.demand for loc in route.locations)
            assert total_demand <= capacity + 1e-9


# ---------------------------------------------------------------------------
# 6. normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    """Invariants for min-max normalization."""

    @given(values=st.lists(float_strategy, min_size=1, max_size=50))
    @settings(max_examples=50, deadline=None)
    def test_output_in_unit_interval(self, values: list[float]) -> None:
        """All output values lie in [0, 1]."""
        result = normalize(values)
        for v in result:
            assert 0.0 <= v <= 1.0

    @given(values=st.lists(float_strategy, min_size=2, max_size=50))
    @settings(max_examples=50, deadline=None)
    def test_preserves_relative_order(self, values: list[float]) -> None:
        """If a < b in input, normalize(a) <= normalize(b) in output."""
        result = normalize(values)
        for i in range(len(values)):
            for j in range(len(values)):
                if values[i] < values[j]:
                    assert result[i] <= result[j] + 1e-12


# ---------------------------------------------------------------------------
# 7. z_score
# ---------------------------------------------------------------------------


class TestZScore:
    """Invariants for z-score computation."""

    @given(
        value=float_strategy,
        mean=float_strategy,
        std=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_sign_matches_deviation_direction(self, value: float, mean: float, std: float) -> None:
        """Sign of z_score matches deviation direction from mean."""
        z = z_score(value, mean, std)
        if value > mean:
            assert z > 0.0
        elif value < mean:
            assert z < 0.0
        else:
            assert z == 0.0
