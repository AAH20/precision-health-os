"""Quality evolution ledger — persistent quality metrics across runs.

Tracks test count, coverage, benchmark pass_rate, and other quality signals
over time so improvement is a measured trend, not a claim. A regression in
any tracked metric fails the gate.

The ledger is append-only: snapshots are never mutated, only added.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Every metric the gate tracks, with its direction.
# higher_is_better=True means a drop is a regression.
# higher_is_better=False means a rise is a regression (e.g. latency).
TRACKED_METRICS: dict[str, dict[str, Any]] = {
    "tests": {"higher_is_better": True, "description": "Total test count"},
    "coverage": {"higher_is_better": True, "description": "Line coverage percent"},
    "benchmark_pass_rate": {
        "higher_is_better": True,
        "description": "Fraction of benchmarks passing",
    },
    "benchmark_mean_score": {"higher_is_better": True, "description": "Mean benchmark score"},
    "ruff_errors": {"higher_is_better": False, "description": "Ruff lint error count"},
    "bandit_high": {"higher_is_better": False, "description": "Bandit high-severity findings"},
    "bandit_medium": {"higher_is_better": False, "description": "Bandit medium-severity findings"},
}


@dataclass(frozen=True)
class QualitySnapshot:
    """A single point-in-time measurement of project quality."""

    label: str
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that all metric names are known."""
        unknown = set(self.metrics) - set(TRACKED_METRICS)
        if unknown:
            raise ValueError(
                f"unknown metric(s): {sorted(unknown)}. Known: {sorted(TRACKED_METRICS)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {"label": self.label, "metrics": dict(self.metrics)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualitySnapshot:
        """Deserialize from a dict produced by to_dict()."""
        return cls(label=data["label"], metrics=dict(data.get("metrics", {})))


class QualityLedger:
    """Append-only history of quality snapshots.

    Persisted as JSON so it survives across CI runs and local invocations.
    The file is the source of truth; the in-memory list is a cache.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        """Initialize the ledger, loading from disk if a path is given."""
        self._path = Path(path) if path else None
        self._snapshots: list[QualitySnapshot] = []
        if self._path and self._path.exists():
            self._load()

    def _load(self) -> None:
        """Load snapshots from disk. Corrupt files yield an empty ledger."""
        path = self._path
        if path is None:
            return
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for entry in raw.get("snapshots", []):
            # Filter unknown metrics BEFORE construction so from_dict's
            # validation doesn't reject forward-incompatible entries.
            metrics = {k: v for k, v in entry.get("metrics", {}).items() if k in TRACKED_METRICS}
            self._snapshots.append(QualitySnapshot(label=entry["label"], metrics=metrics))

    def record(self, snapshot: QualitySnapshot) -> None:
        """Append a snapshot. Never mutates existing entries."""
        self._snapshots.append(snapshot)
        if self._path:
            self.save()

    def save(self) -> None:
        """Persist the ledger to disk."""
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"snapshots": [s.to_dict() for s in self._snapshots]}
        self._path.write_text(json.dumps(payload, indent=2))

    def snapshots(self) -> list[QualitySnapshot]:
        """Return all snapshots in chronological order."""
        return list(self._snapshots)

    def latest(self) -> QualitySnapshot | None:
        """The most recently recorded snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def baseline(self) -> QualitySnapshot | None:
        """The first recorded snapshot — the starting point."""
        return self._snapshots[0] if self._snapshots else None

    def regressions(self, candidate: QualitySnapshot) -> list[dict[str, Any]]:
        """Compare candidate against the best previously recorded value.

        A regression is a drop in a higher-is-better metric (or a rise in a
        lower-is-better metric) relative to the best value seen so far.
        Equality is not a regression.
        """
        if not self._snapshots:
            return []

        # Best value per metric across all prior snapshots.
        best: dict[str, float] = {}
        for snap in self._snapshots:
            for key, val in snap.metrics.items():
                if key not in best:
                    best[key] = val
                elif TRACKED_METRICS[key]["higher_is_better"]:
                    best[key] = max(best[key], val)
                else:
                    best[key] = min(best[key], val)

        results: list[dict[str, Any]] = []
        for key, current in candidate.metrics.items():
            if key not in best:
                continue
            previous = best[key]
            higher_better = TRACKED_METRICS[key]["higher_is_better"]
            if (higher_better and current < previous) or (not higher_better and current > previous):
                results.append(
                    {
                        "metric": key,
                        "previous": previous,
                        "current": current,
                        "delta": current - previous,
                    }
                )
        return results

    def trend(self, metric: str) -> str:
        """Direction of travel for a metric across the full history.

        Returns one of: "improving", "declining", "flat", "unknown".
        Requires at least 2 snapshots to determine a trend.
        """
        if metric not in TRACKED_METRICS:
            raise ValueError(f"unknown metric: {metric}")
        values = [s.metrics[metric] for s in self._snapshots if metric in s.metrics]
        if len(values) < 2:
            return "unknown"
        higher_better = TRACKED_METRICS[metric]["higher_is_better"]
        # Compare first to last.
        first, last = values[0], values[-1]
        if first == last:
            return "flat"
        improved = last > first if higher_better else last < first
        return "improving" if improved else "declining"

    def summary(self) -> dict[str, Any]:
        """Compact human-readable summary of the ledger."""
        trends = {
            m: self.trend(m)
            for m in TRACKED_METRICS
            if any(m in s.metrics for s in self._snapshots)
        }
        latest = self.latest()
        baseline = self.baseline()
        return {
            "snapshots": len(self._snapshots),
            "latest_label": latest.label if latest else None,
            "baseline_label": baseline.label if baseline else None,
            "trends": trends,
        }
