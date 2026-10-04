"""Tests for uncovered lines in evolution.py."""

from __future__ import annotations

from precision_health_os.evolution import QualityLedger, QualitySnapshot

# --- Line 77: _load returns early when path is None ---


def test_load_with_no_path_returns_silently():
    ledger = QualityLedger()
    ledger._load()  # Should not raise


# --- Line 97: save returns early when path is None ---


def test_save_with_no_path_returns_silently():
    ledger = QualityLedger()
    ledger.save()  # Should not raise


# --- Lines 130-133: regressions with lower_is_better metric ---


def test_regressions_lower_is_better_rise_is_regression():
    """A rise in a lower_is_better metric (e.g. ruff_errors) is a regression."""
    ledger = QualityLedger()
    ledger.record(QualitySnapshot(label="v1", metrics={"ruff_errors": 5}))
    candidate = QualitySnapshot(label="v2", metrics={"ruff_errors": 10})
    result = ledger.regressions(candidate)
    assert len(result) == 1
    assert result[0]["metric"] == "ruff_errors"
    assert result[0]["previous"] == 5
    assert result[0]["current"] == 10
    assert result[0]["delta"] == 5


def test_regressions_lower_is_better_best_is_minimum():
    """Best value for lower_is_better is the minimum across snapshots."""
    ledger = QualityLedger()
    ledger.record(QualitySnapshot(label="v1", metrics={"ruff_errors": 10}))
    ledger.record(QualitySnapshot(label="v2", metrics={"ruff_errors": 3}))
    # Candidate at 4: best is 3, 4 > 3 → regression for lower_is_better
    candidate = QualitySnapshot(label="v3", metrics={"ruff_errors": 4})
    result = ledger.regressions(candidate)
    assert len(result) == 1
    assert result[0]["previous"] == 3


# --- Line 138: regressions skips metrics not in any prior snapshot ---


def test_regressions_skips_metric_not_in_prior_snapshots():
    """If a candidate metric was never seen before, it is skipped."""
    ledger = QualityLedger()
    ledger.record(QualitySnapshot(label="v1", metrics={"tests": 100}))
    candidate = QualitySnapshot(label="v2", metrics={"coverage": 95})
    result = ledger.regressions(candidate)
    assert result == []


def test_regressions_unknown_candidate_metric_ignored():
    """Candidate has a metric that no prior snapshot contains."""
    ledger = QualityLedger()
    ledger.record(QualitySnapshot(label="v1", metrics={"tests": 50}))
    candidate = QualitySnapshot(label="v2", metrics={"tests": 60, "bandit_high": 2})
    result = ledger.regressions(candidate)
    # bandit_high is not in any prior snapshot → skipped
    assert all(r["metric"] != "bandit_high" for r in result)


def test_regressions_higher_is_better_best_is_maximum():
    """Best value for higher_is_better is the maximum across snapshots."""
    ledger = QualityLedger()
    ledger.record(QualitySnapshot(label="v1", metrics={"tests": 50}))
    ledger.record(QualitySnapshot(label="v2", metrics={"tests": 80}))
    # Candidate at 70: best is 80, 70 < 80 → regression
    candidate = QualitySnapshot(label="v3", metrics={"tests": 70})
    result = ledger.regressions(candidate)
    assert len(result) == 1
    assert result[0]["previous"] == 80
    assert result[0]["current"] == 70
