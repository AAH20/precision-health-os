# Examples for Precision Health OS
# Basic Usage

from datetime import datetime

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.models import Patient, VitalSigns

api = PrecisionHealthAPI()

# Register patient
patient = Patient(
    id="p001",
    mrn="MRN001",
    name="John Doe",
    date_of_birth=datetime(1990, 1, 1),
    sex="male",
)
api.register_patient(patient)

# Ingest vitals
vitals = VitalSigns(
    patient_id="p001",
    heart_rate_bpm=72,
    spo2_percent=98,
    blood_pressure_systolic=120,
    blood_pressure_diastolic=80,
)
alerts = api.ingest_vitals(vitals)

# Optimization Example
from precision_health_os.optimization import VRPTimeWindowsSolver, Location

depot = Location(id="hospital", x=0, y=0)
locations = [
    Location(id="p1", x=3, y=4, demand=1, service_time=30),
    Location(id="p2", x=-2, y=5, demand=1, service_time=20),
]
solver = VRPTimeWindowsSolver(depot, vehicle_capacity=10)
routes = solver.solve(locations)

# Security Example
import os

from precision_health_os.security import AuditTrail, EncryptionService

key = os.urandom(32)
enc = EncryptionService(key)
ciphertext = enc.encrypt("sensitive data")
plaintext = enc.decrypt(ciphertext)

audit = AuditTrail()
audit.log("user1", "read", "Patient", "p001")
