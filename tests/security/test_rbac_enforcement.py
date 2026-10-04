"""Security tests for RBAC enforcement edge cases."""

from datetime import UTC, datetime

import pytest

from precision_health_os.api import PrecisionHealthAPI
from precision_health_os.models import Patient, VitalSigns
from precision_health_os.security import RBACService


class TestRBACService:
    """Test RBACService role and permission management."""

    def test_admin_role_has_expected_permissions(self):
        rbac = RBACService()
        rbac.assign_role("user1", "admin")
        assert rbac.get_user_permissions("user1") == {"read", "write", "delete", "admin", "audit"}

    def test_physician_role_has_expected_permissions(self):
        rbac = RBACService()
        rbac.assign_role("user1", "physician")
        assert rbac.get_user_permissions("user1") == {"read", "write", "prescribe", "order"}

    def test_nurse_role_has_expected_permissions(self):
        rbac = RBACService()
        rbac.assign_role("user1", "nurse")
        assert rbac.get_user_permissions("user1") == {"read", "write", "administer"}

    def test_researcher_role_has_expected_permissions(self):
        rbac = RBACService()
        rbac.assign_role("user1", "researcher")
        assert rbac.get_user_permissions("user1") == {"read", "export", "anonymize"}

    def test_patient_role_has_expected_permissions(self):
        rbac = RBACService()
        rbac.assign_role("user1", "patient")
        assert rbac.get_user_permissions("user1") == {"read_own", "write_own"}

    def test_permissions_unioned_across_multiple_roles(self):
        rbac = RBACService()
        rbac.assign_role("user1", "nurse")
        rbac.assign_role("user1", "researcher")
        perms = rbac.get_user_permissions("user1")
        assert perms == {"read", "write", "administer", "export", "anonymize"}

    def test_unknown_role_has_no_permissions(self):
        rbac = RBACService()
        assert rbac.get_user_permissions("unknown_user") == set()
        assert not rbac.has_permission("unknown_user", "read")

    def test_check_permission_returns_false_for_unauthorized(self):
        rbac = RBACService()
        rbac.assign_role("user1", "nurse")
        assert not rbac.has_permission("user1", "delete")
        assert not rbac.has_permission("user1", "admin")
        assert not rbac.has_permission("user1", "prescribe")

    def test_assign_unknown_role_raises_error(self):
        rbac = RBACService()
        with pytest.raises(ValueError, match="Unknown role"):
            rbac.assign_role("user1", "superuser")

    def test_user_with_no_roles_has_no_permissions(self):
        rbac = RBACService()
        assert rbac.get_user_permissions("no_roles_user") == set()
        for perm in [
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
        ]:
            assert not rbac.has_permission("no_roles_user", perm)

    def test_user_with_all_roles_has_all_permissions(self):
        rbac = RBACService()
        for role in RBACService.ROLE_PERMISSIONS:
            rbac.assign_role("super_user", role)
        all_perms = set()
        for perms in RBACService.ROLE_PERMISSIONS.values():
            all_perms.update(perms)
        assert rbac.get_user_permissions("super_user") == all_perms


class TestAPIEnforcement:
    """Test that the API enforces RBAC on protected operations."""

    def test_api_check_permission_delegates_to_rbac(self):
        api = PrecisionHealthAPI()
        api.rbac.assign_role("user1", "admin")
        assert api.check_permission("user1", "read")
        assert not api.check_permission("user1", "nonexistent")

    def test_acknowledge_alert_enforces_rbac(self):
        """acknowledge_alert must reject users without proper permissions."""
        api = PrecisionHealthAPI()
        patient = Patient(
            id="p1",
            mrn="MRN001",
            name="Test",
            date_of_birth=datetime(1990, 1, 1, tzinfo=UTC),
            sex="unknown",
        )
        api.register_patient(patient)
        vitals = VitalSigns(patient_id="p1", heart_rate_bpm=200)
        alerts = api.ingest_vitals(vitals)
        if alerts:
            alert_id = alerts[0].id
            result = api.acknowledge_alert(alert_id, "unauthorized_user")
            assert result is False, "BUG: acknowledge_alert does not enforce RBAC"

    def test_get_patient_enforces_rbac(self):
        """get_patient must reject unauthorized access."""
        api = PrecisionHealthAPI()
        patient = Patient(
            id="p1",
            mrn="MRN001",
            name="Test",
            date_of_birth=datetime(1990, 1, 1, tzinfo=UTC),
            sex="unknown",
        )
        api.register_patient(patient)
        result = api.get_patient("p1", user_id="unauthorized_user")
        assert result is None, "BUG: get_patient does not enforce RBAC"
