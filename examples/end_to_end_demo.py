"""End-to-end demonstration of the Precision Health OS platform.

Exercises 11 capabilities: patient registration, vitals ingestion, alerting,
acknowledgement, audit chain verification, anomaly detection, VRPTW, TSP,
pharmacogenomics, and dashboard state building.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.dashboard import build_dashboard_state
from precision_health_os.genomics import PharmacogenomicScorer
from precision_health_os.iot import AnomalyDetector
from precision_health_os.models import Patient, VitalSigns
from precision_health_os.optimization import Location, TSPSolver, VRPTimeWindowsSolver


def section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    """Run the end-to-end demonstration."""
    random.seed(42)
    api = PrecisionHealthAPI()

    # 1. Register a patient
    section("1. Register Patient")
    patient = Patient(
        id="P-1001",
        mrn="MRN-001",
        name="Jane Doe",
        date_of_birth=datetime(1985, 3, 15, tzinfo=UTC),
        sex="female",
        weight_kg=65.0,
        height_cm=168.0,
        allergies=["penicillin"],
        conditions=["hypertension"],
    )
    api.register_patient(patient)
    print(f"  Registered: {patient.name} (ID: {patient.id}, MRN: {patient.mrn})")

    # 2. Ingest normal vitals (no alert expected)
    section("2. Ingest Normal Vitals")
    normal_vitals = VitalSigns(
        patient_id=patient.id,
        heart_rate_bpm=72,
        spo2_percent=98,
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        temperature_celsius=36.6,
    )
    alerts = api.ingest_vitals(normal_vitals)
    print("  Vitals: HR=72, SpO2=98, BP=120/80, Temp=36.6")
    print(f"  Alerts raised: {len(alerts)} (expected 0)")

    # 3. Ingest critical vitals (HR 160, SpO2 82)
    section("3. Ingest Critical Vitals")
    critical_vitals = VitalSigns(
        patient_id=patient.id,
        heart_rate_bpm=160,
        spo2_percent=82,
        blood_pressure_systolic=190,
        blood_pressure_diastolic=110,
        temperature_celsius=39.2,
    )
    alerts = api.ingest_vitals(critical_vitals)
    print("  Vitals: HR=160, SpO2=82, BP=190/110, Temp=39.2")
    print(f"  Alerts raised: {len(alerts)}")
    for a in alerts:
        print(f"    [{a.severity.value.upper()}] {a.title}: {a.description}")

    # 4. Show active alerts, acknowledge, show shrink
    section("4. Acknowledge Alerts")
    active = api.get_active_alerts(patient_id=patient.id)
    print(f"  Active alerts before ack: {len(active)}")
    for a in active:
        api.acknowledge_alert(a.id, user_id="dr.smith")
        print(f"    Acknowledged: {a.title}")
    active_after = api.get_active_alerts(patient_id=patient.id)
    print(f"  Active alerts after ack: {len(active_after)}")

    # 5. Verify audit chain
    section("5. Verify Audit Chain")
    chain_ok = api.audit.verify_chain()
    print(f"  Audit chain intact: {chain_ok}")
    print(f"  Total audit events: {len(api.audit._events)}")

    # 6. Anomaly detection on HR series with spike
    section("6. Anomaly Detection (Ensemble)")
    detector = AnomalyDetector()
    hr_series = [72, 74, 71, 73, 75, 72, 70, 73, 74, 180, 72, 71]
    results = detector.detect_ensemble(hr_series)
    spike_idx = 9
    spike = results[spike_idx]
    print(f"  Series: {hr_series}")
    print(f"  Spike at index {spike_idx} (value={hr_series[spike_idx]})")
    print(f"  Ensemble verdict: is_anomaly={spike.is_anomaly}, score={spike.score:.2f}")
    print(f"  Votes: {spike.details['votes']}/{spike.details['voters']}")

    # 7. Solve a small VRPTW instance
    section("7. VRPTW Route Optimization")
    depot = Location(id="depot", x=0.0, y=0.0)
    locations = [
        Location(id="L1", x=2.0, y=3.0, demand=1.0, ready_time=0.0, due_time=10.0),
        Location(id="L2", x=5.0, y=1.0, demand=1.0, ready_time=0.0, due_time=12.0),
        Location(id="L3", x=1.0, y=6.0, demand=1.0, ready_time=0.0, due_time=15.0),
        Location(id="L4", x=7.0, y=4.0, demand=1.0, ready_time=0.0, due_time=20.0),
    ]
    vrp_solver = VRPTimeWindowsSolver(depot=depot, vehicle_capacity=10.0)
    routes = vrp_solver.solve(locations)
    print(f"  Locations: {len(locations)}, Vehicles used: {len(routes)}")
    for i, route in enumerate(routes):
        loc_ids = [loc.id for loc in route.locations]
        print(f"    Route {i + 1}: {' -> '.join(loc_ids)} (dist={route.total_distance:.2f})")

    # 8. Solve a small TSP instance
    section("8. TSP Tour Optimization")
    tsp_solver = TSPSolver()
    tsp_locations = [
        Location(id="A", x=0.0, y=0.0),
        Location(id="B", x=3.0, y=4.0),
        Location(id="C", x=6.0, y=0.0),
        Location(id="D", x=3.0, y=1.0),
    ]
    tour = tsp_solver.solve(tsp_locations)
    tour_ids = [loc.id for loc in tour.locations]
    print(f"  Tour: {' -> '.join(tour_ids)}")
    print(f"  Tour length: {tour.total_distance:.2f}")

    # 9. Pharmacogenomic scoring
    section("9. Pharmacogenomic Scoring")
    scorer = PharmacogenomicScorer()
    guideline = scorer.score_drug("CYP2D6", "*4/*4", "codeine")
    if guideline:
        print(f"  Gene: {guideline.gene}, Genotype: {guideline.allele}")
        print(f"  Phenotype: {guideline.phenotype}")
        print(f"  Drug: {guideline.drug}")
        print(f"  Recommendation: {guideline.recommendation}")
        print(f"  Evidence level: {guideline.evidence_level}")
    else:
        print("  No guideline found")

    # 10. Build dashboard state
    section("10. Dashboard State")
    dashboard = build_dashboard_state(api)
    print(f"  Top-level keys: {list(dashboard.keys())}")
    print(f"  System status: {dashboard['system_status']}")
    print(f"  Modules tracked: {len(dashboard['modules'])}")

    # 11. Final summary
    section("11. Summary")
    status = api.get_system_status()
    print(
        f"  Demo complete: {status['patients_registered']} patient(s), "
        f"{status['active_alerts']} active alert(s), "
        f"{status['audit_events']} audit event(s), "
        f"{status['event_bus_events']} event-bus event(s)."
    )


if __name__ == "__main__":
    main()
