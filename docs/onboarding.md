# Onboarding & Sizing Guide

## Deployment Tiers

### Clinic Tier (1-50 patients)

- **Hardware**: 2 vCPU, 4GB RAM, 50GB SSD
- **Database**: SQLite or single PostgreSQL instance
- **Redis**: Optional (in-memory queue)
- **Use Case**: Single practice, basic CDSS, patient scheduling

### Hospital Tier (50-10,000 patients)

- **Hardware**: 8 vCPU, 32GB RAM, 500GB SSD
- **Database**: PostgreSQL 16 with read replicas
- **Redis**: Required (caching + queue)
- **Workers**: 2-4 background workers
- **Use Case**: Multi-department, full CDSS, genomics, imaging

### Health System Tier (10,000+ patients)

- **Hardware**: 16+ vCPU, 64GB+ RAM, 1TB+ NVMe
- **Database**: PostgreSQL cluster with sharding
- **Redis**: Redis Cluster
- **Workers**: 8+ background workers
- **GPU**: NVIDIA A100 for ML inference
- **Use Case**: Multi-site, federated learning, drug discovery

## Module Sizing

| Module | Clinic | Hospital | Health System |
|--------|--------|----------|---------------|
| CDSS | ✓ | ✓ | ✓ |
| Genomics | - | ✓ | ✓ |
| Imaging AI | - | GPU | GPU Cluster |
| Wearables | Basic | Full | Full + Edge |
| Drug Discovery | - | - | ✓ |
| FHIR/HL7 | FHIR | FHIR + HL7 | Full Interop |

## Scaling Considerations

- **Database**: Use read replicas for analytics queries
- **Caching**: Redis for session data and frequent queries
- **Queue**: Celery + Redis for background tasks
- **ML Inference**: Separate GPU nodes for model serving
- **FHIR**: Use FHIR bulk data export for large datasets
