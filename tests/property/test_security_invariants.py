"""Property-based invariant tests for the security module."""

from __future__ import annotations

import os

from hypothesis import given, settings
from hypothesis import strategies as st

from precision_health_os.security import (
    AuditTrail,
    EncryptionService,
    HIPAACompliance,
    RBACService,
)

# Strategies
non_empty_text = st.text(min_size=1, max_size=200)
valid_outcomes = st.sampled_from(["success", "failure"])
valid_roles = st.sampled_from(["admin", "physician", "nurse", "researcher", "patient"])
valid_permissions = st.sampled_from(
    [
        "read",
        "write",
        "delete",
        "admin",
        "audit",
        "prescribe",
        "order",
        "administer",
        "export",
        "anonymize",
        "read_own",
        "write_own",
    ]
)


class TestEncryptionService:
    """Property-based tests for EncryptionService."""

    @given(plaintext=non_empty_text)
    @settings(max_examples=30, deadline=None)
    def test_decrypt_encrypt_roundtrip(self, plaintext: str) -> None:
        """decrypt(encrypt(x)) == x for any non-empty text."""
        key = os.urandom(32)
        service = EncryptionService(key)
        ciphertext = service.encrypt(plaintext)
        assert service.decrypt(ciphertext) == plaintext

    @given(plaintext=non_empty_text)
    @settings(max_examples=30, deadline=None)
    def test_nonce_uniqueness(self, plaintext: str) -> None:
        """Two encryptions of same text differ (nonce uniqueness)."""
        key = os.urandom(32)
        service = EncryptionService(key)
        ct1 = service.encrypt(plaintext)
        ct2 = service.encrypt(plaintext)
        assert ct1 != ct2

    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_encrypt_dict_roundtrip(self, data: dict) -> None:
        """decrypt_dict(encrypt_dict(d)) == d for any JSON-serializable dict."""
        key = os.urandom(32)
        service = EncryptionService(key)
        ciphertext = service.encrypt_dict(data)
        assert service.decrypt_dict(ciphertext) == data


class TestAuditTrail:
    """Property-based tests for AuditTrail."""

    @given(
        user_id=st.text(min_size=1, max_size=30),
        action=st.text(min_size=1, max_size=30),
        resource_type=st.text(min_size=1, max_size=30),
        resource_id=st.text(min_size=1, max_size=30),
        outcome=valid_outcomes,
    )
    @settings(max_examples=30, deadline=None)
    def test_verify_chain_after_single_event(
        self, user_id, action, resource_type, resource_id, outcome
    ) -> None:
        """verify_chain() is True after logging a single event."""
        trail = AuditTrail()
        trail.log(user_id, action, resource_type, resource_id, outcome)
        assert trail.verify_chain() is True

    @given(
        events=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=30),
                st.text(min_size=1, max_size=30),
                st.text(min_size=1, max_size=30),
                st.text(min_size=1, max_size=30),
                valid_outcomes,
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_verify_chain_after_multiple_events(self, events) -> None:
        """verify_chain() is True after any sequence of logged events."""
        trail = AuditTrail()
        for user_id, action, resource_type, resource_id, outcome in events:
            trail.log(user_id, action, resource_type, resource_id, outcome)
        assert trail.verify_chain() is True

    @given(
        events=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=30),
                st.text(min_size=1, max_size=30),
                st.text(min_size=1, max_size=30),
                st.text(min_size=1, max_size=30),
                valid_outcomes,
            ),
            min_size=2,
            max_size=10,
        ),
        mutate_index=st.integers(min_value=0, max_value=9),
    )
    @settings(max_examples=30, deadline=None)
    def test_verify_chain_false_after_mutation(self, events, mutate_index) -> None:
        """verify_chain() is False after mutating any single event."""
        trail = AuditTrail()
        for user_id, action, resource_type, resource_id, outcome in events:
            trail.log(user_id, action, resource_type, resource_id, outcome)

        idx = mutate_index % len(events)
        event = trail._events[idx]
        event.action = event.action + "_MUTATED"

        assert trail.verify_chain() is False


class TestRBACService:
    """Property-based tests for RBACService."""

    @given(
        user_id=st.text(min_size=1, max_size=30),
        permission=st.text(min_size=1, max_size=30),
    )
    @settings(max_examples=30, deadline=None)
    def test_has_permission_returns_bool(self, user_id: str, permission: str) -> None:
        """has_permission returns bool for any user/permission."""
        service = RBACService()
        result = service.has_permission(user_id, permission)
        assert isinstance(result, bool)

    @given(
        user_id=st.text(min_size=1, max_size=30),
        role=valid_roles,
        permission=valid_permissions,
    )
    @settings(max_examples=30, deadline=None)
    def test_has_permission_with_role(self, user_id, role, permission) -> None:
        """has_permission returns True when user has role with that permission."""
        service = RBACService()
        service.assign_role(user_id, role)
        result = service.has_permission(user_id, permission)
        assert isinstance(result, bool)
        if permission in RBACService.ROLE_PERMISSIONS[role]:
            assert result is True


class TestHIPAACompliance:
    """Property-based tests for HIPAACompliance."""

    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=30),
            values=st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_deidentify_removes_safe_harbor_fields(self, data: dict) -> None:
        """Deidentify removes all 18 Safe Harbor identifier types."""
        result = HIPAACompliance.deidentify(data)
        for field in HIPAACompliance.SAFE_HARBOR_FIELDS:
            assert field.lower() not in {k.lower() for k in result}

    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=30),
            values=st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_is_deidentified_after_deidentify(self, data: dict) -> None:
        """is_deidentified returns True after deidentify."""
        result = HIPAACompliance.deidentify(data)
        assert HIPAACompliance.is_deidentified(result) is True

    @given(
        data=st.dictionaries(
            keys=st.sampled_from(list(HIPAACompliance.SAFE_HARBOR_FIELDS)),
            values=st.text(max_size=50),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_deidentify_removes_all_safe_harbor_fields(self, data: dict) -> None:
        """Deidentify returns empty dict when all keys are Safe Harbor fields."""
        result = HIPAACompliance.deidentify(data)
        assert len(result) == 0
