# System Overview Architecture

```mermaid
flowchart TB
    subgraph Edge["Edge Layer"]
        Wearables[Wearable Sensors<br/>CGM, ECG, SpO2]
        IoMT[IoMT Devices<br/>Infusion Pumps, Ventilators]
        Imaging[Imaging Devices<br/>CT, MRI, X-ray, Pathology]
    end

    subgraph Ingestion["Ingestion Layer"]
        GW[API Gateway<br/>REST + WebSocket]
        FHIR[FHIR/HL7 Interop]
        Stream[Event Stream<br/>Kafka/Redis]
    end

    subgraph Core["Core Platform"]
        subgraph Optimization["Optimization Engine"]
            VRP[Patient Scheduling<br/>VRP w/ Time Windows]
            TSP[Treatment Planning<br/>TSP Variants]
            KA[Resource Allocation<br/>Knapsack]
            BM[Trial Matching<br/>Bipartite Matching]
        end

        subgraph AI["AI/ML Engine"]
            DP[Disease Prediction]
            DR[Drug Response Modeling]
            MI[Medical Image Analysis]
            AD[Anomaly Detection]
        end

        subgraph Clinical["Clinical Engine"]
            CDSS[CDSS Rules Engine]
            PATH[Clinical Pathway Optimizer]
            ALERT[Alert Management]
        end

        subgraph Genomics["Genomics Engine"]
            VC[Variant Calling]
            PGx[Pharmacogenomic Scoring]
            RP[Risk Prediction]
        end

        subgraph DrugDisc["Drug Discovery"]
            DOCK[Molecular Docking]
            ADMET[ADMET Prediction]
            CTM[Clinical Trial Matching]
        end
    end

    subgraph Infrastructure["Infrastructure"]
        SEC[Security Layer<br/>HIPAA, Encryption, Audit]
        DB[(PostgreSQL<br/>FHIR Resources)]
        CACHE[(Redis<br/>Session + Cache)]
        QUEUE[Task Queue<br/>Celery/Redis]
    end

    subgraph External["External Systems"]
        EHR[EHR Systems<br/>Epic, Cerner, OpenMRS]
        LIS[LIS/LIMS]
        TRIAL[Clinical Trial Registries]
    end

    Wearables --> Stream
    IoMT --> Stream
    Imaging --> GW
    Stream --> GW
    GW --> FHIR
    FHIR --> Core
    EHR <--> FHIR
    LIS <--> GW
    TRIAL <--> GW
    Core --> DB
    Core --> CACHE
    Core --> QUEUE
    SEC -.-> Core
    SEC -.-> DB
    SEC -.-> GW
```
