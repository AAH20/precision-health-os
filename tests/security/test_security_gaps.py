"""Gap tests for precision_health_os.security uncovered lines."""

from datetime import UTC, datetime, timedelta

import pytest
from cryptography.exceptions import InvalidTag

import precision_health_os.security as security_mod
from precision_health_os.security import (
    AuditTrail,
    EncryptionService,
    HIPAACompliance,
    KeyRotationService,
    RBACService,
)

MASTER_KEY = b"\x01" * 32


@pytest.fixture
def fixed_clock(monkeypatch):
    """Controllable fake clock: call base() to read/reset the current time."""
    state = {"now": datetime(2026, 1, 1, tzinfo=UTC)}

    def utcnow():
        return state["now"]

    def base(**kwargs):
        state["now"] = state["now"] + timedelta(**kwargs)
        return state["now"]

    monkeypatch.setattr(security_mod, "utcnow", utcnow)
    return base


# --- EncryptionService.decrypt: tampered ciphertext raises (line 42) ---


def test_decrypt_tampered_ciphertext_raises():
    svc = EncryptionService(MASTER_KEY)
    ct = bytearray(svc.encrypt("secret"))
    ct[-1] ^= 0xFF  # flip bits in last byte of ciphertext (auth tag region)
    with pytest.raises(InvalidTag):
        svc.decrypt(bytes(ct))


def test_decrypt_tampered_nonce_raises():
    svc = EncryptionService(MASTER_KEY)
    ct = bytearray(svc.encrypt("secret"))
    ct[0] ^= 0x01  # corrupt the nonce
    with pytest.raises(InvalidTag):
        svc.decrypt(bytes(ct))


# --- KeyRotationService.get_key: unknown key_id raises (line 73) ---


def test_get_key_unknown_id_raises_keyerror():
    svc = KeyRotationService()
    with pytest.raises(KeyError):
        svc.get_key("no-such-key-id")


# --- KeyRotationService.get_current_key_id: no key yet (line 79) ---


def test_get_current_key_id_before_any_key_raises():
    svc = KeyRotationService()
    with pytest.raises(RuntimeError):
        svc.get_current_key_id()


# --- AuditTrail.get_events: date-range filtering (lines 150, 152) ---


def test_get_events_start_filter(fixed_clock):
    trail = AuditTrail()
    trail.log("u1", "view", "patient", "p1")
    trail.log("u2", "edit", "patient", "p2")
    trail.log("u3", "view", "patient", "p3")
    fixed_clock(hours=1)  # boundary: everything before this is excluded
    e4 = trail.log("u4", "view", "patient", "p4")
    res = trail.get_events(start=e4.timestamp)
    assert [e.resource_id for e in res] == ["p4"]
    # earlier events fall outside the window
    cutoff = e4.timestamp - timedelta(seconds=1)
    assert [e.resource_id for e in trail.get_events(end=cutoff)] == ["p1", "p2", "p3"]


def test_get_events_end_filter(fixed_clock):
    trail = AuditTrail()
    trail.log("u1", "view", "patient", "p1")
    e2 = trail.log("u2", "view", "patient", "p2")
    res = trail.get_events(end=e2.timestamp)
    assert [e.resource_id for e in res] == ["p1", "p2"]


def test_get_events_combined_user_and_date_filter(fixed_clock):
    trail = AuditTrail()
    trail.log("alice", "view", "patient", "p1")
    trail.log("bob", "view", "patient", "p2")
    trail.log("alice", "view", "patient", "p3")
    fixed_clock(days=1)  # boundary
    e4 = trail.log("alice", "view", "patient", "p4")
    res = trail.get_events(user_id="alice", start=e4.timestamp)
    assert [e.resource_id for e in res] == ["p4"]


# --- RBACService.get_user_permissions: user with no roles (line 184) ---


def test_get_user_permissions_no_roles_returns_empty_set():
    rbac = RBACService()
    assert rbac.get_user_permissions("nobody") == set()


# --- HIPAACompliance.is_deidentified: edge cases (line 225) ---


def test_is_deidentified_empty_dict():
    assert HIPAACompliance.is_deidentified({}) is True


def test_is_deidentified_case_insensitive_field():
    assert HIPAACompliance.is_deidentified({"NAME": "Alice"}) is False
