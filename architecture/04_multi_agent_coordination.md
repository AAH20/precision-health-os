# Multi-Agent Coordination Sequence

```mermaid
sequenceDiagram
    participant Pat as Patient
    participant GW as API Gateway
    participant RPM as RPM Engine
    participant OPT as Optimizer
    participant CDSS as CDSS Engine
    participant GEN as Genomics
    participant DD as Drug Discovery
    participant SEC as Security
    participant DB as Database

    Pat->>GW: Wearable data (HR, SpO2, Glucose)
    GW->>SEC: Authenticate + Authorize
    SEC->>DB: Fetch patient context
    DB-->>SEC: Patient record (FHIR)
    SEC->>RPM: Forward validated data
    RPM->>RPM: Anomaly detection (Z-score + LSTM)
    
    alt Anomaly detected
        RPM->>CDSS: Trigger clinical alert
        CDSS->>CDSS: Rule evaluation (Drools-like)
        CDSS->>GEN: Query pharmacogenomic profile
        GEN->>DB: Fetch variant data
        DB-->>GEN: Variants + PGx score
        GEN-->>CDSS: Drug-gene interactions
        
        CDSS->>DD: Query alternative therapies
        DD->>DD: ADMET prediction + docking
        DD-->>CDSS: Ranked treatment options
        
        CDSS->>OPT: Optimize treatment schedule
        OPT->>OPT: VRP solve (patient routing)
        OPT-->>CDSS: Optimized schedule
        
        CDSS->>SEC: Log decision + audit trail
        CDSS-->>GW: Alert + recommendations
        GW-->>Pat: Notification (push/SMS)
    end
    
    Note over RPM,DB: Continuous monitoring loop
    loop Every 60 seconds
        Pat->>GW: New sensor reading
        GW->>RPM: Ingest reading
        RPM->>RPM: Update digital twin
    end
```
