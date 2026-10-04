"""Tests for security module."""

import os

import pytest

from precision_health_os.security import (
    AuditTrail,
    EncryptionService,
    HIPAACompliance,
    KeyRotationService,
    RBACService,
)


class TestEncryptionService:
    """Tests for EncryptionService."""

    def setup_method(self) -> None:
        self.key = os.urandom(32)
        self.service = EncryptionService(self.key)

    def test_encrypt_decrypt(self) -> None:
        plaintext = "sensitive patient data"
        ciphertext = self.service.encrypt(plaintext)
        decrypted = self.service.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_encrypt_dict(self) -> None:
        data = {"name": "John", "ssn": "123-45-6789"}
        ciphertext = self.service.encrypt_dict(data)
        decrypted = self.service.decrypt_dict(ciphertext)
        assert decrypted == data

    def test_different_ciphertexts(self) -> None:
        plaintext = "test"
        ct1 = self.service.encrypt(plaintext)
        ct2 = self.service.encrypt(plaintext)
        assert ct1 != ct2  # Different nonces

    def test_invalid_key_size(self) -> None:
        with pytest.raises(ValueError):
            EncryptionService(b"short_key")


class TestKeyRotationService:
    """Tests for KeyRotationService."""

    def test_generate_key(self) -> None:
        service = KeyRotationService()
        key_id = service.generate_key()
        assert key_id is not None
        assert service.get_current_key_id() == key_id

    def test_rotate(self) -> None:
        service = KeyRotationService()
        old_id = service.generate_key()
        new_id = service.rotate()
        assert new_id != old_id
        assert service.get_current_key_id() == new_id

    def test_get_key(self) -> None:
        service = KeyRotationService()
        key_id = service.generate_key()
        key = service.get_key(key_id)
        assert len(key) == 32


class TestAuditTrail:
    """Tests for AuditTrail."""

    def test_log_event(self) -> None:
        trail = AuditTrail()
        event = trail.log("user1", "read", "Patient", "p001")
        assert event.user_id == "user1"
        assert event.action == "read"

    def test_chain_integrity(self) -> None:
        trail = AuditTrail()
        trail.log("user1", "read", "Patient", "p001")
        trail.log("user2", "write", "Observation", "o001")
        assert trail.verify_chain()

    def test_tamper_detection(self) -> None:
        trail = AuditTrail()
        trail.log("user1", "read", "Patient", "p001")
        # Tamper with the event
        trail._events[0].action = "delete"
        assert not trail.verify_chain()

    def test_query_events(self) -> None:
        trail = AuditTrail()
        trail.log("user1", "read", "Patient", "p001")
        trail.log("user2", "read", "Patient", "p002")
        trail.log("user1", "write", "Observation", "o001")

        user1_events = trail.get_events(user_id="user1")
        assert len(user1_events) == 2

        patient_events = trail.get_events(resource_type="Patient")
        assert len(patient_events) == 2


class TestRBACService:
    """Tests for RBACService."""

    def test_assign_role(self) -> None:
        rbac = RBACService()
        rbac.assign_role("user1", "physician")
        assert rbac.has_permission("user1", "read")
        assert rbac.has_permission("user1", "write")
        assert not rbac.has_permission("user1", "admin")

    def test_multiple_roles(self) -> None:
        rbac = RBACService()
        rbac.assign_role("user1", "physician")
        rbac.assign_role("user1", "researcher")
        assert rbac.has_permission("user1", "read")
        assert rbac.has_permission("user1", "export")

    def test_invalid_role(self) -> None:
        rbac = RBACService()
        with pytest.raises(ValueError):
            rbac.assign_role("user1", "invalid_role")

    def test_get_permissions(self) -> None:
        rbac = RBACService()
        rbac.assign_role("user1", "nurse")
        perms = rbac.get_user_permissions("user1")
        assert "read" in perms
        assert "write" in perms


class TestHIPAACompliance:
    """Tests for HIPAACompliance."""

    def test_deidentify(self) -> None:
        data = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "diagnosis": "diabetes",
            "age": 45,
        }
        result = HIPAACompliance.deidentify(data)
        assert "name" not in result
        assert "ssn" not in result
        assert "diagnosis" in result
        assert "age" in result

    def test_is_deidentified(self) -> None:
        clean_data = {"diagnosis": "diabetes", "age": 45}
        dirty_data = {"name": "John", "diagnosis": "diabetes"}
        assert HIPAACompliance.is_deidentified(clean_data)
        assert not HIPAACompliance.is_deidentified(dirty_data)
