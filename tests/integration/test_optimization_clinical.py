"""Integration tests: optimization → clinical decision support."""

from datetime import UTC, datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.clinical import CDSSEngine
from precision_health_os.models import AlertSeverity, Patient, VitalSigns
from precision_health_os.optimization import (
    BipartiteMatchingSolver,
    KnapsackSolver,
    Location,
    TSPSolver,
    VRPTimeWindowsSolver,
)

# ── VRPTW Tests ──


def test_vrptw_routes_are_feasible():
    depot = Location(id="depot", x=0, y=0)
    locations = [
        Location(id="p1", x=3, y=4, demand=1, ready_time=0, due_time=50, service_time=5),
        Location(id="p2", x=6, y=8, demand=1, ready_time=0, due_time=50, service_time=5),
        Location(id="p3", x=1, y=1, demand=1, ready_time=0, due_time=50, service_time=5),
        Location(id="p4", x=5, y=2, demand=1, ready_time=0, due_time=50, service_time=5),
    ]
    solver = VRPTimeWindowsSolver(depot=depot, vehicle_capacity=10)
    routes = solver.solve(locations)
    assert len(routes) > 0
    covered = [loc.id for r in routes for loc in r.locations]
    assert set(covered) == {loc.id for loc in locations}
    for route in routes:
        assert route.feasible


def test_vrptw_respects_capacity():
    depot = Location(id="depot", x=0, y=0)
    locations = [
        Location(id=f"p{i}", x=float(i), y=0, demand=3, ready_time=0, due_time=100, service_time=1)
        for i in range(5)
    ]
    solver = VRPTimeWindowsSolver(depot=depot, vehicle_capacity=7)
    routes = solver.solve(locations)
    for route in routes:
        assert sum(loc.demand for loc in route.locations) <= 7


# ── TSP Tests ──


def test_tsp_tour_visits_all_locations():
    locations = [
        Location(id="a", x=0, y=0),
        Location(id="b", x=3, y=4),
        Location(id="c", x=6, y=0),
        Location(id="d", x=3, y=1),
    ]
    solver = TSPSolver()
    route = solver.solve(locations)
    assert len(route.locations) == len(locations)
    visited_ids = [loc.id for loc in route.locations]
    assert set(visited_ids) == {loc.id for loc in locations}
    assert len(visited_ids) == len(set(visited_ids))


def test_tsp_tour_distance_is_reasonable():
    locations = [
        Location(id="a", x=0, y=0),
        Location(id="b", x=10, y=0),
        Location(id="c", x=5, y=8),
        Location(id="d", x=3, y=3),
    ]
    solver = TSPSolver()
    route = solver.solve(locations)
    naive_dist = sum(
        ((locations[i].x - locations[i + 1].x) ** 2 + (locations[i].y - locations[i + 1].y) ** 2)
        ** 0.5
        for i in range(len(locations) - 1)
    )
    assert route.total_distance <= naive_dist + 1e-9


# ── Knapsack Tests ──


def test_knapsack_solution_is_optimal():
    items = [
        {"name": "A", "value": 60, "weight": 10},
        {"name": "B", "value": 100, "weight": 20},
        {"name": "C", "value": 120, "weight": 30},
    ]
    solver = KnapsackSolver(capacity=50)
    selected = solver.solve(items)
    assert sum(item["weight"] for item in selected) <= 50
    assert sum(item["value"] for item in selected) == 220


def test_knapsack_respects_capacity():
    items = [
        {"name": "A", "value": 10, "weight": 5},
        {"name": "B", "value": 20, "weight": 10},
        {"name": "C", "value": 30, "weight": 15},
    ]
    solver = KnapsackSolver(capacity=12)
    selected = solver.solve(items)
    assert sum(item["weight"] for item in selected) <= 12


# ── Bipartite Matching Tests ──


def test_bipartite_matching_valid():
    patients = [
        {"id": "p1", "conditions": ["diabetes"], "age": 50},
        {"id": "p2", "conditions": ["cancer"], "age": 60},
    ]
    trials = [
        {"id": "t1", "conditions": ["diabetes"], "min_age": 18, "max_age": 65},
        {"id": "t2", "conditions": ["cancer"], "min_age": 18, "max_age": 70},
    ]
    solver = BipartiteMatchingSolver()
    matches = solver.solve(patients, trials)
    assert len(matches) == 2
    matched_patients = [m[0] for m in matches]
    matched_trials = [m[1] for m in matches]
    assert len(set(matched_patients)) == len(matched_patients)
    assert len(set(matched_trials)) == len(matched_trials)
    for _, _, score in matches:
        assert score > 0


def test_bipartite_matching_respects_compatibility():
    patients = [
        {"id": "p1", "conditions": ["diabetes"], "age": 50},
        {"id": "p2", "conditions": ["asthma"], "age": 30},
    ]
    trials = [{"id": "t1", "conditions": ["diabetes"], "min_age": 18, "max_age": 65}]
    solver = BipartiteMatchingSolver()
    matches = solver.solve(patients, trials)
    assert len(matches) == 1
    assert matches[0][0] == "p1"
    assert matches[0][1] == "t1"


# ── Optimization → Clinical Integration ──


def test_optimization_results_inform_clinical_decisions():
    drugs = [
        {"name": "metformin", "value": 80, "weight": 5},
        {"name": "insulin", "value": 90, "weight": 8},
        {"name": "glipizide", "value": 60, "weight": 4},
    ]
    knapsack = KnapsackSolver(capacity=10)
    selected_drugs = knapsack.solve(drugs)
    engine = CDSSEngine()
    patient = Patient(
        id="p1",
        mrn="MRN001",
        name="Test Patient",
        date_of_birth=datetime(1970, 1, 1, tzinfo=UTC),
        sex="male",
        conditions=["diabetes"],
    )
    context = {"selected_drugs": [d["name"] for d in selected_drugs]}
    alerts = engine.evaluate_rules(patient, context)
    assert isinstance(alerts, list)


def test_vrptw_routes_used_for_patient_scheduling():
    depot = Location(id="clinic", x=0, y=0)
    patients = [
        Location(id="p1", x=2, y=3, demand=1, ready_time=0, due_time=30, service_time=10),
        Location(id="p2", x=5, y=1, demand=1, ready_time=0, due_time=30, service_time=10),
        Location(id="p3", x=1, y=5, demand=1, ready_time=0, due_time=30, service_time=10),
    ]
    solver = VRPTimeWindowsSolver(depot=depot, vehicle_capacity=5)
    routes = solver.solve(patients)
    for route in routes:
        assert len(route.locations) > 0
        time = 0.0
        prev = depot
        for loc in route.locations:
            dist = ((prev.x - loc.x) ** 2 + (prev.y - loc.y) ** 2) ** 0.5
            arrival = time + dist
            assert arrival <= loc.due_time
            time = max(arrival, loc.ready_time) + loc.service_time
            prev = loc


# ── Full Flow Tests ──


def test_full_flow_register_optimize_verify():
    api = PrecisionHealthAPI()
    patients = []
    for i in range(4):
        p = Patient(
            id=f"patient-{i}",
            mrn=f"MRN{i:03d}",
            name=f"Patient {i}",
            date_of_birth=datetime(1980 + i, 1, 1, tzinfo=UTC),
            sex="male" if i % 2 == 0 else "female",
            conditions=["hypertension"] if i % 2 == 0 else ["diabetes"],
        )
        api.register_patient(p)
        patients.append(p)
    assert api.get_system_status()["patients_registered"] == 4
    depot = Location(id="hospital", x=0, y=0)
    locations = [
        Location(
            id=p.id,
            x=float(i * 3),
            y=float(i * 2),
            demand=1,
            ready_time=0,
            due_time=100,
            service_time=15,
        )
        for i, p in enumerate(patients)
    ]
    solver = VRPTimeWindowsSolver(depot=depot, vehicle_capacity=10)
    routes = solver.solve(locations)
    assert len(routes) > 0
    covered = [loc.id for r in routes for loc in r.locations]
    assert set(covered) == {p.id for p in patients}
    for route in routes:
        assert route.feasible
        assert sum(loc.demand for loc in route.locations) <= 10


def test_full_flow_with_clinical_alerts():
    api = PrecisionHealthAPI()
    patient = Patient(
        id="p1",
        mrn="MRN001",
        name="Test",
        date_of_birth=datetime(1970, 1, 1, tzinfo=UTC),
        sex="male",
        conditions=["diabetes"],
    )
    api.register_patient(patient)
    vitals = VitalSigns(
        patient_id="p1", heart_rate_bpm=160, blood_pressure_systolic=210, spo2_percent=88
    )
    alerts = api.ingest_vitals(vitals)
    assert len(alerts) > 0
    assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)
    assert len(api.get_active_alerts(patient_id="p1")) > 0


# ── Edge Cases ──


def test_vrptw_empty_input():
    depot = Location(id="depot", x=0, y=0)
    assert VRPTimeWindowsSolver(depot=depot).solve([]) == []


def test_tsp_empty_input():
    route = TSPSolver().solve([])
    assert route.locations == []


def test_tsp_single_location():
    route = TSPSolver().solve([Location(id="a", x=1, y=1)])
    assert len(route.locations) == 1
    assert route.locations[0].id == "a"


def test_knapsack_empty_input():
    assert KnapsackSolver(capacity=10).solve([]) == []


def test_knapsack_single_item_fits():
    selected = KnapsackSolver(capacity=10).solve([{"name": "A", "value": 50, "weight": 5}])
    assert len(selected) == 1
    assert selected[0]["name"] == "A"


def test_knapsack_single_item_too_heavy():
    assert KnapsackSolver(capacity=3).solve([{"name": "A", "value": 50, "weight": 5}]) == []


def test_bipartite_empty_patients():
    assert BipartiteMatchingSolver().solve([], [{"id": "t1", "conditions": ["cancer"]}]) == []


def test_bipartite_empty_trials():
    assert BipartiteMatchingSolver().solve([{"id": "p1", "conditions": ["cancer"]}], []) == []
