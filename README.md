# Precision Health OS

Unified precision health platform integrating genomics, clinical decision support, medical imaging, wearables, drug discovery, and health IoT.

## Architecture

See [architecture/](architecture/) for system diagrams.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest --cov=src/precision_health_os

# Start server
phos serve --host 0.0.0.0 --port 8000

# Docker
docker compose up -d
```

## Modules

| Module | Purpose |
|--------|---------|
| `optimization` | NP-hard solvers: VRP, TSP, Knapsack, Bipartite Matching |
| `clinical` | CDSS engine, alert management, pathway optimization |
| `ml` | Disease prediction, drug response, medical imaging |
| `iot` | Wearable pipeline, anomaly detection, remote monitoring |
| `genomics` | Variant calling, pharmacogenomic scoring, risk prediction |
| `drug_discovery` | Molecular docking, ADMET, trial matching |
| `security` | HIPAA compliance, encryption, audit trails, RBAC |
| `integration` | FHIR/HL7, EHR connector, event bus |
| `api` | REST API facade |
| `models` | Pydantic data models |

## NP-Hard Problems Solved

| Problem | Complexity | Solver |
|---------|-----------|--------|
| Patient Scheduling (VRPTW) | O(n²·2ⁿ) | Clarke-Wright heuristic |
| Treatment Planning (TSP) | O(n!) | NN + 2-opt |
| Drug Combination (Knapsack) | Pseudo-poly | DP |
| Trial Matching (Bipartite) | O(V·E) | Greedy matching |

## License

MIT
