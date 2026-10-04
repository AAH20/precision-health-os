"""Core data models for Precision Health OS."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Patient(BaseModel):
    """Patient demographic and clinical context."""

    id: str = Field(..., description="Unique patient identifier")
    mrn: str = Field(..., description="Medical record number")
    name: str
    date_of_birth: datetime
    sex: str = Field(..., pattern="^(male|female|other|unknown)$")
    weight_kg: float | None = Field(None, gt=0)
    height_cm: float | None = Field(None, gt=0)
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VitalSigns(BaseModel):
    """Real-time vital signs from wearables or IoMT devices."""

    patient_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    heart_rate_bpm: float | None = Field(None, ge=0, le=300)
    blood_pressure_systolic: float | None = Field(None, ge=0, le=300)
    blood_pressure_diastolic: float | None = Field(None, ge=0, le=200)
    spo2_percent: float | None = Field(None, ge=0, le=100)
    temperature_celsius: float | None = Field(None, ge=30, le=45)
    respiratory_rate: float | None = Field(None, ge=0, le=100)
    glucose_mg_dl: float | None = Field(None, ge=0, le=1000)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ClinicalAlert(BaseModel):
    """Clinical decision support alert."""

    id: str
    patient_id: str
    severity: AlertSeverity
    title: str
    description: str
    source: str = Field(..., description="Rule or model that generated the alert")
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommendations: list[str] = Field(default_factory=list)
    acknowledged: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GenomicVariant(BaseModel):
    """Genomic variant with clinical significance."""

    chrom: str
    pos: int = Field(..., gt=0)
    ref: str
    alt: str
    gene: str | None = None
    consequence: str | None = None
    clinical_significance: str | None = None
    allele_frequency: float | None = Field(None, ge=0, le=1)
    zygosity: str = Field("heterozygous", pattern="^(homozygous|heterozygous|hemizygous)$")


class DrugResponsePrediction(BaseModel):
    """Predicted drug response for a patient."""

    patient_id: str
    drug_name: str
    predicted_response: str = Field(..., pattern="^(responder|non-responder|intermediate)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    biomarkers: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)


class TreatmentPlan(BaseModel):
    """Optimized treatment plan."""

    patient_id: str
    diagnosis: str
    medications: list[dict[str, Any]] = Field(default_factory=list)
    procedures: list[dict[str, Any]] = Field(default_factory=list)
    schedule: list[datetime] = Field(default_factory=list)
    duration_days: int = Field(..., gt=0)
    expected_outcome: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)


class FHIRResource(BaseModel):
    """FHIR R4 resource wrapper."""

    resource_type: str
    id: str | None = None
    meta: dict[str, Any] | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class AuditEvent(BaseModel):
    """HIPAA-compliant audit event."""

    id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    outcome: str = Field(..., pattern="^(success|failure)$")
    ip_address: str | None = None
    user_agent: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
