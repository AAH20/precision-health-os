"""Gap-coverage tests for precision_health_os.clinical.

Targets: vital-threshold boundaries, custom rule evaluation, the safe AST
evaluator (including adversarial inputs), alert management/escalation,
and clinical pathway optimization.
"""

import ast
from datetime import UTC, datetime

import pytest

from precision_health_os.clinical import (
    AlertManager,
    CDSSEngine,
    ClinicalPathwayOptimizer,
    ClinicalRule,
)
from precision_health_os.models import AlertSeverity, ClinicalAlert, Patient, VitalSigns
from precision_health_os.utils import generate_id

NOW = datetime(2026, 10, 4, tzinfo=UTC)


def make_patient(**kw):
    d = dict(
        id="p1",
        mrn="MRN1",
        name="Test Patient",
        date_of_birth=datetime(1980, 1, 1, tzinfo=UTC),
        sex="male",
    )
    d.update(kw)
    return Patient(**d)


def make_vitals(**kw):
    d = dict(
        patient_id="p1",
        heart_rate_bpm=72,
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        spo2_percent=98,
        temperature_celsius=37.0,
        respiratory_rate=16,
        glucose_mg_dl=100,
    )
    d.update(kw)
    return VitalSigns(**d)


def make_alert(patient_id="p1", source="s", title="t", severity=AlertSeverity.HIGH, **kw):
    d = dict(
        id=generate_id(),
        patient_id=patient_id,
        severity=severity,
        title=title,
        description="d",
        source=source,
        confidence=0.9,
    )
    d.update(kw)
    return ClinicalAlert(**d)


def make_rule(rid="r1", condition="age > 65", **kw):
    d = dict(
        id=rid,
        name="rule",
        condition=condition,
        severity=AlertSeverity.HIGH,
        message="msg",
        recommendations=["rec"],
    )
    d.update(kw)
    return ClinicalRule(**d)


class TestVitalThresholds:
    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("heart_rate_bpm", 25, AlertSeverity.CRITICAL),
            ("heart_rate_bpm", 30, AlertSeverity.HIGH),
            ("heart_rate_bpm", 39, AlertSeverity.HIGH),
            ("heart_rate_bpm", 40, None),
            ("heart_rate_bpm", 72, None),
            ("heart_rate_bpm", 120, None),
            ("heart_rate_bpm", 121, AlertSeverity.HIGH),
            ("heart_rate_bpm", 150, AlertSeverity.HIGH),
            ("heart_rate_bpm", 151, AlertSeverity.CRITICAL),
            ("spo2_percent", 84, AlertSeverity.CRITICAL),
            ("spo2_percent", 85, AlertSeverity.HIGH),
            ("spo2_percent", 91, AlertSeverity.HIGH),
            ("spo2_percent", 92, None),
            ("spo2_percent", 98, None),
            ("blood_pressure_systolic", 89, AlertSeverity.HIGH),
            ("blood_pressure_systolic", 90, None),
            ("blood_pressure_systolic", 181, AlertSeverity.HIGH),
            ("blood_pressure_systolic", 180, None),
            ("blood_pressure_systolic", 200, AlertSeverity.HIGH),
            ("blood_pressure_systolic", 201, AlertSeverity.CRITICAL),
            ("blood_pressure_diastolic", 59, AlertSeverity.HIGH),
            ("blood_pressure_diastolic", 60, None),
            ("blood_pressure_diastolic", 111, AlertSeverity.HIGH),
            ("blood_pressure_diastolic", 110, None),
            ("blood_pressure_diastolic", 130, AlertSeverity.HIGH),
            ("blood_pressure_diastolic", 131, AlertSeverity.CRITICAL),
            ("temperature_celsius", 34.9, AlertSeverity.HIGH),
            ("temperature_celsius", 35.0, None),
            ("temperature_celsius", 38.6, AlertSeverity.HIGH),
            ("temperature_celsius", 38.5, None),
            ("temperature_celsius", 40.0, AlertSeverity.HIGH),
            ("temperature_celsius", 40.1, AlertSeverity.CRITICAL),
            ("respiratory_rate", 7, AlertSeverity.HIGH),
            ("respiratory_rate", 8, None),
            ("respiratory_rate", 25, AlertSeverity.HIGH),
            ("respiratory_rate", 24, None),
            ("respiratory_rate", 35, AlertSeverity.HIGH),
            ("respiratory_rate", 36, AlertSeverity.CRITICAL),
            ("glucose_mg_dl", 49, AlertSeverity.CRITICAL),
            ("glucose_mg_dl", 50, AlertSeverity.HIGH),
            ("glucose_mg_dl", 69, AlertSeverity.HIGH),
            ("glucose_mg_dl", 70, None),
            ("glucose_mg_dl", 181, AlertSeverity.HIGH),
            ("glucose_mg_dl", 180, None),
            ("glucose_mg_dl", 300, AlertSeverity.HIGH),
            ("glucose_mg_dl", 301, AlertSeverity.CRITICAL),
        ],
    )
    def test_threshold_boundaries(self, field, value, expected):
        alerts = CDSSEngine().evaluate_vitals("p1", make_vitals(**{field: value}))
        if expected is None:
            assert alerts == []
        else:
            assert len(alerts) == 1
            assert alerts[0].severity == expected
            assert alerts[0].patient_id == "p1"

    def test_none_vitals_skipped(self):
        assert CDSSEngine().evaluate_vitals("p1", make_vitals(heart_rate_bpm=None)) == []

    def test_multiple_simultaneous_alerts(self):
        alerts = CDSSEngine().evaluate_vitals(
            "p1", make_vitals(heart_rate_bpm=200, spo2_percent=80, glucose_mg_dl=350)
        )
        assert len(alerts) == 3
        assert all(a.severity == AlertSeverity.CRITICAL for a in alerts)

    def test_alert_metadata(self):
        crit = CDSSEngine().evaluate_vitals("p1", make_vitals(heart_rate_bpm=25))
        assert crit[0].confidence == 0.95
        assert crit[0].source == "vital_thresholds"
        assert crit[0].recommendations == [
            "Immediate clinical assessment required",
            "Consider emergency intervention",
        ]
        low = CDSSEngine().evaluate_vitals("p1", make_vitals(heart_rate_bpm=39))
        assert low[0].confidence == 0.85
        assert low[0].recommendations == ["Monitor closely", "Consider further evaluation"]


class TestCustomRules:
    def test_firing_rule_alert_fields(self):
        engine = CDSSEngine()
        engine.add_rule(make_rule())
        patient = make_patient(date_of_birth=datetime(1950, 1, 1, tzinfo=UTC))
        alerts = engine.evaluate_rules(patient, {"now": NOW})
        assert len(alerts) == 1
        a = alerts[0]
        assert (
            a.patient_id,
            a.severity,
            a.title,
            a.description,
            a.source,
            a.confidence,
            a.recommendations,
        ) == ("p1", AlertSeverity.HIGH, "rule", "msg", "rule:r1", 0.8, ["rec"])

    def test_non_firing_and_disabled_rules(self):
        engine = CDSSEngine()
        engine.add_rule(make_rule(enabled=False))
        patient = make_patient(date_of_birth=datetime(1950, 1, 1, tzinfo=UTC))
        assert engine.evaluate_rules(patient, {"now": NOW}) == []
        engine.add_rule(make_rule(rid="r2", condition="age > 200"))
        assert engine.evaluate_rules(patient, {"now": NOW}) == []

    def test_patient_attribute_and_binop_conditions(self):
        engine = CDSSEngine()
        engine.add_rule(make_rule(rid="r1", condition="patient.weight_kg > 100"))
        engine.add_rule(make_rule(rid="r2", condition="age > 60 and age < 70"))
        patient = make_patient(weight_kg=120, date_of_birth=datetime(1966, 1, 1, tzinfo=UTC))
        alerts = engine.evaluate_rules(patient, {"now": NOW})
        assert {a.source for a in alerts} == {"rule:r1", "rule:r2"}

    def test_condition_error_not_raised(self):
        # _evaluate_condition swallows all exceptions internally, so the
        # except branch in evaluate_rules (line 151-152) is unreachable.
        engine = CDSSEngine()
        engine.add_rule(make_rule(condition="1/0"))
        assert engine.evaluate_rules(make_patient(), {}) == []

    def test_multiple_rules_and_remove(self):
        engine = CDSSEngine()
        engine.add_rule(make_rule(rid="r1"))
        engine.add_rule(
            make_rule(rid="r2", severity=AlertSeverity.LOW, condition="patient.sex == 'male'")
        )
        patient = make_patient(date_of_birth=datetime(1950, 1, 1, tzinfo=UTC))
        assert len(engine.evaluate_rules(patient, {"now": NOW})) == 2
        engine.remove_rule("r1")
        engine.remove_rule("unknown")
        alerts = engine.evaluate_rules(patient, {"now": NOW})
        assert len(alerts) == 1 and alerts[0].source == "rule:r2"


class TestEvaluateCondition:
    def setup_method(self):
        self.engine = CDSSEngine()
        self.patient = make_patient(date_of_birth=datetime(1950, 1, 1, tzinfo=UTC))

    def test_age_computed_with_now(self):
        assert self.engine._evaluate_condition("age > 65", self.patient, {"now": NOW}) is True

    def test_age_zero_without_now(self):
        assert self.engine._evaluate_condition("age > 65", self.patient, {}) is False

    def test_invalid_syntax_returns_false(self):
        assert self.engine._evaluate_condition("1 +", self.patient, {}) is False

    def test_context_values_available(self):
        assert self.engine._evaluate_condition("risk > 0.5", self.patient, {"risk": 0.7}) is True


class TestSafeEvalNode:
    def setup_method(self):
        self.engine = CDSSEngine()

    def eval_expr(self, expr, namespace=None):
        tree = ast.parse(expr, mode="eval")
        return self.engine._safe_eval_node(tree.body, namespace or {})

    def test_constant_and_name(self):
        assert self.eval_expr("42") == 42
        assert self.eval_expr("x", {"x": 5}) == 5
        assert self.eval_expr("missing") is False

    def test_binop(self):
        assert self.eval_expr("1 + 2") == 3
        assert self.eval_expr("5 - 3") == 2
        assert self.eval_expr("3 * 4") == 12
        assert self.eval_expr("10 / 4") == 2.5

    def test_unaryop(self):
        assert self.eval_expr("-5") == -5
        assert self.eval_expr("+5") == 5
        assert self.eval_expr("not True") is False

    def test_boolop(self):
        assert self.eval_expr("True and False") is False
        assert self.eval_expr("False or True") is True

    def test_compare(self):
        assert self.eval_expr("1 < 2") is True
        assert self.eval_expr("2 <= 2") is True
        assert self.eval_expr("3 > 2") is True
        assert self.eval_expr("2 >= 3") is False
        assert self.eval_expr("2 == 2") is True
        assert self.eval_expr("2 != 3") is True

    def test_chained_compare(self):
        assert self.eval_expr("1 < 2 < 3") is True
        assert self.eval_expr("1 < 2 > 3") is False
        assert self.eval_expr("1 < 5 > 3") is True

    def test_attribute(self):
        patient = make_patient()
        assert self.eval_expr("patient.name", {"patient": patient}) == "Test Patient"
        assert self.eval_expr("patient.nope", {"patient": patient}) is None

    def test_call(self):
        assert self.eval_expr("f(3)", {"f": lambda x: x * 2}) == 6
        assert self.eval_expr("x(1)", {"x": 5}) is False

    def test_in_and_not_in(self):
        assert self.eval_expr("2 in vals", {"vals": [1, 2, 3]}) is True
        assert self.eval_expr("5 not in vals", {"vals": [1, 2, 3]}) is True
        assert self.eval_expr("5 in vals", {"vals": [1, 2, 3]}) is False

    def test_is_and_is_not(self):
        assert self.eval_expr("x is None", {"x": None}) is True
        assert self.eval_expr("x is not None", {"x": 1}) is True

    def test_unknown_node_is_false(self):
        assert self.engine._safe_eval_node(ast.parse("lambda: 1", mode="eval").body, {}) is False
        assert self.engine._safe_eval_node(ast.parse("[1, 2][0]", mode="eval").body, {}) is False


class TestAdversarialAst:
    def setup_method(self):
        self.engine = CDSSEngine()
        self.patient = make_patient()

    def test_import_rejected(self):
        assert (
            self.engine._evaluate_condition("__import__('os').system('ls')", self.patient, {})
            is False
        )

    def test_eval_exec_open_rejected(self):
        assert self.engine._evaluate_condition("eval('1+1')", self.patient, {}) is False
        assert self.engine._evaluate_condition("exec('print(1)')", self.patient, {}) is False
        assert (
            self.engine._evaluate_condition("open('/etc/passwd').read()", self.patient, {}) is False
        )

    def test_dunder_names_not_in_namespace(self):
        assert self.engine._evaluate_condition("__builtins__", self.patient, {}) is False
        assert self.engine._evaluate_condition("__import__", self.patient, {}) is False

    def test_dunder_class_walk_broken_at_subscript(self):
        # Subscript node is unhandled -> chain stops, nothing executes
        assert (
            self.engine._evaluate_condition(
                "patient.__class__.__bases__[0].__subclasses__()", self.patient, {}
            )
            is False
        )

    def test_dunder_getattr_is_inert(self):
        # getattr yields read-only values; no code execution.
        # _evaluate_condition coerces to bool, so 'Patient' -> True.
        assert (
            self.engine._evaluate_condition("patient.__class__.__name__", self.patient, {}) is True
        )

    def test_malicious_rule_condition_inert(self):
        engine = CDSSEngine()
        engine.add_rule(make_rule(condition="__import__('os').system('ls')"))
        assert engine.evaluate_rules(self.patient, {}) == []


class TestAlertManager:
    def test_add_and_dedup(self):
        mgr = AlertManager()
        a1 = make_alert()
        assert mgr.add_alert(a1) is a1
        assert mgr.add_alert(make_alert()) is a1  # same patient/source/title
        assert mgr.add_alert(make_alert(title="b")).title == "b"
        assert mgr.add_alert(make_alert(source="s2")).source == "s2"
        assert mgr.get_alert_stats() == {"high": 3}

    def test_acknowledge_and_unknown(self):
        mgr = AlertManager()
        a = mgr.add_alert(make_alert())
        assert mgr.acknowledge(a.id, "dr") is a
        assert a.acknowledged is True
        assert mgr.acknowledge("nope", "dr") is None

    def test_dedup_skips_acknowledged(self):
        mgr = AlertManager()
        a1 = mgr.add_alert(make_alert(created_at=datetime(2026, 1, 1, tzinfo=UTC)))
        mgr.acknowledge(a1.id, "dr")
        a2 = mgr.add_alert(make_alert(created_at=datetime(2026, 1, 2, tzinfo=UTC)))
        assert a2 is not a1
        a3 = make_alert(created_at=datetime(2026, 1, 3, tzinfo=UTC))
        assert mgr.add_alert(a3) is a2  # loops past acknowledged a1

    def test_active_alerts_filters(self):
        mgr = AlertManager()
        a1 = mgr.add_alert(make_alert(patient_id="p1", title="a"))
        mgr.add_alert(make_alert(patient_id="p2", title="b"))
        mgr.add_alert(make_alert(patient_id="p1", title="c", severity=AlertSeverity.LOW))
        mgr.acknowledge(a1.id, "dr")
        assert len(mgr.get_active_alerts()) == 2
        assert len(mgr.get_active_alerts(patient_id="p1")) == 1
        assert len(mgr.get_active_alerts(patient_id="p2")) == 1
        low = mgr.get_active_alerts(severity=AlertSeverity.LOW)
        assert len(low) == 1 and low[0].severity == AlertSeverity.LOW

    def test_active_alerts_sorted_by_created_at_desc(self):
        mgr = AlertManager()
        mgr.add_alert(make_alert(title="a1", created_at=datetime(2026, 1, 1, tzinfo=UTC)))
        mgr.add_alert(make_alert(title="a2", created_at=datetime(2026, 1, 3, tzinfo=UTC)))
        mgr.add_alert(make_alert(title="a3", created_at=datetime(2026, 1, 2, tzinfo=UTC)))
        assert [a.title for a in mgr.get_active_alerts()] == ["a2", "a3", "a1"]

    def test_stats_multiple_severities(self) -> None:
        """AlertManager deduplicates identical patient/source/title alerts, so
        distinct severities must differ in title to be counted separately.
        """
        mgr = AlertManager()
        mgr.add_alert(make_alert(severity=AlertSeverity.CRITICAL, title="crit-a"))
        mgr.add_alert(make_alert(severity=AlertSeverity.CRITICAL, title="crit-b"))
        mgr.add_alert(make_alert(severity=AlertSeverity.LOW, title="low-a"))
        assert mgr.get_alert_stats() == {"critical": 2, "low": 1}

    def test_identical_alerts_are_deduplicated_not_counted(self) -> None:
        """The counterpart guarantee: repeats collapse to one alert."""
        mgr = AlertManager()
        mgr.add_alert(make_alert(severity=AlertSeverity.CRITICAL, title="same"))
        mgr.add_alert(make_alert(severity=AlertSeverity.CRITICAL, title="same"))
        assert mgr.get_alert_stats() == {"critical": 1}


class TestPathwayOptimizer:
    def test_unknown_pathway_raises(self):
        with pytest.raises(ValueError, match="Unknown pathway"):
            ClinicalPathwayOptimizer().optimize("nope", {})

    def test_register_and_optimize(self):
        opt = ClinicalPathwayOptimizer()
        opt.register_pathway("sepsis", [{"id": "s1"}, {"id": "s2"}])
        assert [s["id"] for s in opt.optimize("sepsis", {})] == ["s1", "s2"]

    def test_optimize_filters_by_condition(self):
        opt = ClinicalPathwayOptimizer()
        opt.register_pathway(
            "p",
            [
                {"id": "a", "condition": "risk > 0.5"},
                {"id": "b", "condition": "risk > 0.9"},
                {"id": "c"},
            ],
        )
        assert [s["id"] for s in opt.optimize("p", {"risk": 0.7})] == ["a", "c"]
        assert [s["id"] for s in opt.optimize("p", {"risk": 0.1})] == ["c"]

    def test_condition_error_step_applicable(self):
        opt = ClinicalPathwayOptimizer()
        opt.register_pathway("p", [{"id": "x", "condition": "1 +"}])
        assert [s["id"] for s in opt.optimize("p", {})] == ["x"]

    def test_empty_pathway(self):
        opt = ClinicalPathwayOptimizer()
        opt.register_pathway("empty", [])
        assert opt.optimize("empty", {}) == []
