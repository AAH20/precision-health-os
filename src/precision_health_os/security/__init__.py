"""Security module: HIPAA compliance, encryption, and audit trails."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from precision_health_os.models import AuditEvent
from precision_health_os.utils import generate_id, utcnow

logger = logging.getLogger(__name__)


class EncryptionService:
    """AES-256-GCM encryption for PHI/PII data."""

    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes for AES-256")
        self._master_key = master_key

    def encrypt(self, plaintext: str, associated_data: bytes | None = None) -> bytes:
        """Encrypt plaintext with AES-256-GCM."""
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        aesgcm = AESGCM(self._master_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), associated_data)
        return nonce + ciphertext

    def decrypt(self, ciphertext: bytes, associated_data: bytes | None = None) -> str:
        """Decrypt AES-256-GCM ciphertext."""
        nonce, encrypted = ciphertext[:12], ciphertext[12:]
        aesgcm = AESGCM(self._master_key)
        plaintext = aesgcm.decrypt(nonce, encrypted, associated_data)
        return plaintext.decode()

    def encrypt_dict(self, data: dict[str, Any]) -> bytes:
        """Encrypt a dictionary as JSON."""
        return self.encrypt(json.dumps(data))

    def decrypt_dict(self, ciphertext: bytes) -> dict[str, Any]:
        """Decrypt and parse JSON."""
        return json.loads(self.decrypt(ciphertext))


class KeyRotationService:
    """Manage encryption key rotation."""

    def __init__(self) -> None:
        self._keys: dict[str, tuple[bytes, datetime]] = {}
        self._current_key_id: str | None = None

    def generate_key(self) -> str:
        """Generate a new encryption key."""
        key_id = generate_id()
        key = AESGCM.generate_key(bit_length=256)
        self._keys[key_id] = (key, utcnow())
        self._current_key_id = key_id
        return key_id

    def get_key(self, key_id: str) -> bytes:
        """Retrieve a key by ID."""
        if key_id not in self._keys:
            raise KeyError(f"Unknown key ID: {key_id}")
        return self._keys[key_id][0]

    def get_current_key_id(self) -> str:
        """Get the current active key ID."""
        if self._current_key_id is None:
            raise RuntimeError("No active key. Call generate_key() first.")
        return self._current_key_id

    def rotate(self) -> str:
        """Rotate to a new key, keeping old keys for decryption."""
        return self.generate_key()


class AuditTrail:
    """HIPAA-compliant immutable audit trail."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._chain_hash: str = "0" * 64

    def log(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        outcome: str = "success",
        ip_address: str | None = None,
        user_agent: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log an audit event with hash chain integrity."""
        event = AuditEvent(
            id=generate_id(),
            timestamp=utcnow(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
        )
        event_data = f"{event.id}:{event.timestamp.isoformat()}:{user_id}:{action}:{resource_id}"
        self._chain_hash = hashlib.sha256(
            f"{self._chain_hash}:{event_data}".encode()
        ).hexdigest()
        self._events.append(event)
        logger.info(f"Audit: {action} on {resource_type}/{resource_id} by {user_id}")
        return event

    def verify_chain(self) -> bool:
        """Verify the integrity of the audit chain."""
        chain_hash = "0" * 64
        for event in self._events:
            event_data = f"{event.id}:{event.timestamp.isoformat()}:{event.user_id}:{event.action}:{event.resource_id}"
            chain_hash = hashlib.sha256(
                f"{chain_hash}:{event_data}".encode()
            ).hexdigest()
        return chain_hash == self._chain_hash

    def get_events(
        self,
        user_id: str | None = None,
        resource_type: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[AuditEvent]:
        """Query audit events with filters."""
        events = self._events
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if resource_type:
            events = [e for e in events if e.resource_type == resource_type]
        if start:
            events = [e for e in events if e.timestamp >= start]
        if end:
            events = [e for e in events if e.timestamp <= end]
        return events


class RBACService:
    """Role-based access control for healthcare."""

    ROLE_PERMISSIONS: dict[str, set[str]] = {
        "admin": {"read", "write", "delete", "admin", "audit"},
        "physician": {"read", "write", "prescribe", "order"},
        "nurse": {"read", "write", "administer"},
        "researcher": {"read", "export", "anonymize"},
        "patient": {"read_own", "write_own"},
    }

    def __init__(self) -> None:
        self._user_roles: dict[str, set[str]] = {}

    def assign_role(self, user_id: str, role: str) -> None:
        """Assign a role to a user."""
        if role not in self.ROLE_PERMISSIONS:
            raise ValueError(f"Unknown role: {role}")
        self._user_roles.setdefault(user_id, set()).add(role)

    def has_permission(self, user_id: str, permission: str) -> bool:
        """Check if a user has a specific permission."""
        roles = self._user_roles.get(user_id, set())
        for role in roles:
            if permission in self.ROLE_PERMISSIONS.get(role, set()):
                return True
        return False

    def get_user_permissions(self, user_id: str) -> set[str]:
        """Get all permissions for a user."""
        roles = self._user_roles.get(user_id, set())
        perms: set[str] = set()
        for role in roles:
            perms.update(self.ROLE_PERMISSIONS.get(role, set()))
        return perms


class HIPAACompliance:
    """HIPAA Safe Harbor de-identification."""

    SAFE_HARBOR_FIELDS = {
        "name", "address", "dates", "telephone", "fax", "email",
        "ssn", "mrn", "health_plan", "account", "certificate",
        "vehicle", "device", "url", "ip", "biometric", "photo",
        "unique_id",
    }

    @staticmethod
    def deidentify(data: dict[str, Any]) -> dict[str, Any]:
        """Remove HIPAA Safe Harbor identifiers."""
        return {
            k: v for k, v in data.items()
            if k.lower() not in HIPAACompliance.SAFE_HARBOR_FIELDS
        }

    @staticmethod
    def is_deidentified(data: dict[str, Any]) -> bool:
        """Check if data has been de-identified."""
        return not any(
            k.lower() in HIPAACompliance.SAFE_HARBOR_FIELDS
            for k in data
        )
