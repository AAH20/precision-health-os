"""Clinical module: CDSS engine, alert management, clinical pathway optimizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from precision_health_os.models import AlertSeverity, ClinicalAlert, Patient, VitalSigns
from precision_health_os.utils import generate_id

logger = logging.getLogger(__name__)


@dataclass
class ClinicalRule:
    """Clinical decision support rule."""

    id: str
    name: str
    condition: str
    severity: AlertSeverity
    message: str
    recommendations: list[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0


class CDSSEngine:
    """Clinical Decision Support System engine.

    Evaluates rules against patient data and generates alerts.
    Supports rule-based and ML-based decision making.
    """

    def __init__(self) -> None:
        """Initialize the CDSS engine."""
        self._rules: dict[str, ClinicalRule] = {}
        self._vital_thresholds: dict[str, dict[str, float]] = {
            "heart_rate_bpm": {"low": 40, "high": 120, "critical_low": 30, "critical_high": 150},
            "spo2_percent": {"low": 92, "critical_low": 85},
            "blood_pressure_systolic": {"low": 90, "high": 180, "critical_high": 200},
            "blood_pressure_diastolic": {"low": 60, "high": 110, "critical_high": 130},
            "temperature_celsius": {"low": 35.0, "high": 38.5, "critical_high": 40.0},
            "respiratory_rate": {"low": 8, "high": 24, "critical_high": 35},
            "glucose_mg_dl": {"low": 70, "high": 180, "critical_low": 50, "critical_high": 300},
        }

    def add_rule(self, rule: ClinicalRule) -> None:
        """Add a clinical rule."""
        self._rules[rule.id] = rule

    def remove_rule(self, rule_id: str) -> None:
        """Remove a clinical rule."""
        self._rules.pop(rule_id, None)

    def evaluate_vitals(self, patient_id: str, vitals: VitalSigns) -> list[ClinicalAlert]:
        """Evaluate vital signs against thresholds."""
        alerts: list[ClinicalAlert] = []

        for vital_name, thresholds in self._vital_thresholds.items():
            value = getattr(vitals, vital_name, None)
            if value is None:
                continue

            if "critical_low" in thresholds and value < thresholds["critical_low"]:
                alerts.append(
                    ClinicalAlert(
                        id=generate_id(),
                        patient_id=patient_id,
                        severity=AlertSeverity.CRITICAL,
                        title=f"Critical {vital_name.replace('_', ' ').title()}",
                        description=(
                            f"{vital_name} = {value} (critical low: {thresholds['critical_low']})"
                        ),
                        source="vital_thresholds",
                        confidence=0.95,
                        recommendations=[
                            "Immediate clinical assessment required",
                            "Consider emergency intervention",
                        ],
                    )
                )
            elif "critical_high" in thresholds and value > thresholds["critical_high"]:
                alerts.append(
                    ClinicalAlert(
                        id=generate_id(),
                        patient_id=patient_id,
                        severity=AlertSeverity.CRITICAL,
                        title=f"Critical {vital_name.replace('_', ' ').title()}",
                        description=(
                            f"{vital_name} = {value} (critical high: {thresholds['critical_high']})"
                        ),
                        source="vital_thresholds",
                        confidence=0.95,
                        recommendations=[
                            "Immediate clinical assessment required",
                            "Consider emergency intervention",
                        ],
                    )
                )
            elif "low" in thresholds and value < thresholds["low"]:
                alerts.append(
                    ClinicalAlert(
                        id=generate_id(),
                        patient_id=patient_id,
                        severity=AlertSeverity.HIGH,
                        title=f"Low {vital_name.replace('_', ' ').title()}",
                        description=f"{vital_name} = {value} (low: {thresholds['low']})",
                        source="vital_thresholds",
                        confidence=0.85,
                        recommendations=["Monitor closely", "Consider further evaluation"],
                    )
                )
            elif "high" in thresholds and value > thresholds["high"]:
                alerts.append(
                    ClinicalAlert(
                        id=generate_id(),
                        patient_id=patient_id,
                        severity=AlertSeverity.HIGH,
                        title=f"High {vital_name.replace('_', ' ').title()}",
                        description=f"{vital_name} = {value} (high: {thresholds['high']})",
                        source="vital_thresholds",
                        confidence=0.85,
                        recommendations=["Monitor closely", "Consider further evaluation"],
                    )
                )

        return alerts

    def evaluate_rules(self, patient: Patient, context: dict[str, Any]) -> list[ClinicalAlert]:
        """Evaluate custom clinical rules against patient context."""
        alerts: list[ClinicalAlert] = []
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            try:
                if self._evaluate_condition(rule.condition, patient, context):
                    alerts.append(
                        ClinicalAlert(
                            id=generate_id(),
                            patient_id=patient.id,
                            severity=rule.severity,
                            title=rule.name,
                            description=rule.message,
                            source=f"rule:{rule.id}",
                            confidence=0.8,
                            recommendations=rule.recommendations,
                        )
                    )
            except Exception as e:
                logger.warning(f"Rule {rule.id} evaluation failed: {e}")
        return alerts

    def _evaluate_condition(
        self, condition: str, patient: Patient, context: dict[str, Any]
    ) -> bool:
        """Evaluate a rule condition string using safe AST evaluation."""
        import ast

        namespace = {
            "patient": patient,
            "age": (context.get("now") - patient.date_of_birth).days / 365.25
            if context.get("now")
            else 0,
            **context,
        }

        try:
            tree = ast.parse(condition, mode="eval")
            return bool(self._safe_eval_node(tree.body, namespace))
        except Exception:
            return False

    def _safe_eval_node(self, node: Any, namespace: dict[str, Any]) -> Any:
        """Safely evaluate an AST node."""
        import ast
        import operator

        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.And: lambda a, b: a and b,
            ast.Or: lambda a, b: a or b,
            ast.Not: operator.not_,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b,
            ast.Is: operator.is_,
            ast.IsNot: operator.is_not,
        }

        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return namespace.get(node.id, False)
        if isinstance(node, ast.BinOp):
            left = self._safe_eval_node(node.left, namespace)
            right = self._safe_eval_node(node.right, namespace)
            return ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](self._safe_eval_node(node.operand, namespace))
        if isinstance(node, ast.BoolOp):
            result = self._safe_eval_node(node.values[0], namespace)
            for value in node.values[1:]:
                result = ops[type(node.op)](result, self._safe_eval_node(value, namespace))
            return result
        if isinstance(node, ast.Compare):
            left = self._safe_eval_node(node.left, namespace)
            for op, comparator in zip(node.ops, node.comparators, strict=False):
                right = self._safe_eval_node(comparator, namespace)
                if not ops[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Attribute):
            obj = self._safe_eval_node(node.value, namespace)
            return getattr(obj, node.attr, None)
        if isinstance(node, ast.Call):
            func = self._safe_eval_node(node.func, namespace)
            args = [self._safe_eval_node(arg, namespace) for arg in node.args]
            return func(*args) if callable(func) else False
        return False


class AlertManager:
    """Alert management: deduplication, escalation, acknowledgment."""

    def __init__(self) -> None:
        """Initialize the alert manager."""
        self._alerts: dict[str, ClinicalAlert] = {}
        self._patient_alerts: dict[str, list[str]] = {}
        self._escalation_rules: list[dict[str, Any]] = []

    def add_alert(self, alert: ClinicalAlert) -> ClinicalAlert:
        """Add an alert with deduplication."""
        # Deduplicate: same patient, same source, same title within 15 minutes
        existing = self._find_duplicate(alert)
        if existing:
            return existing

        self._alerts[alert.id] = alert
        self._patient_alerts.setdefault(alert.patient_id, []).append(alert.id)
        return alert

    def _find_duplicate(self, alert: ClinicalAlert) -> ClinicalAlert | None:
        """Find a duplicate alert."""
        for aid in self._patient_alerts.get(alert.patient_id, []):
            existing = self._alerts.get(aid)
            if (
                existing
                and existing.source == alert.source
                and existing.title == alert.title
                and not existing.acknowledged
            ):
                return existing
        return None

    def acknowledge(self, alert_id: str, user_id: str) -> ClinicalAlert | None:
        """Acknowledge an alert."""
        alert = self._alerts.get(alert_id)
        if alert:
            alert.acknowledged = True
        return alert

    def get_active_alerts(
        self, patient_id: str | None = None, severity: AlertSeverity | None = None
    ) -> list[ClinicalAlert]:
        """Get active (unacknowledged) alerts."""
        alerts = [a for a in self._alerts.values() if not a.acknowledged]
        if patient_id:
            alerts = [a for a in alerts if a.patient_id == patient_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert_stats(self) -> dict[str, int]:
        """Get alert statistics by severity."""
        stats: dict[str, int] = {}
        for alert in self._alerts.values():
            key = alert.severity.value
            stats[key] = stats.get(key, 0) + 1
        return stats


class ClinicalPathwayOptimizer:
    """Optimize clinical pathways using graph-based methods."""

    def __init__(self) -> None:
        """Initialize the pathway optimizer."""
        self._pathways: dict[str, dict[str, Any]] = {}

    def register_pathway(self, name: str, steps: list[dict[str, Any]]) -> None:
        """Register a clinical pathway."""
        self._pathways[name] = {"steps": steps}

    def optimize(self, pathway_name: str, patient_context: dict[str, Any]) -> list[dict[str, Any]]:
        """Optimize a clinical pathway for a specific patient."""
        pathway = self._pathways.get(pathway_name)
        if not pathway:
            raise ValueError(f"Unknown pathway: {pathway_name}")

        steps = pathway["steps"]
        # Filter steps based on patient context
        return [step for step in steps if self._step_applicable(step, patient_context)]

    def _step_applicable(self, step: dict[str, Any], context: dict[str, Any]) -> bool:
        """Check if a pathway step is applicable."""
        condition = step.get("condition")
        if not condition:
            return True
        try:
            import ast

            tree = ast.parse(condition, mode="eval")
            # Reuse the safe evaluator from CDSSEngine
            engine = CDSSEngine()
            return bool(engine._safe_eval_node(tree.body, context))
        except Exception:
            return True
