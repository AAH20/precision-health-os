"""Comprehensive tests for EvolutionTracker.

Covers empty/single-entry safety, monotonic improvement, regression detection,
plateau behaviour, recovery, history immutability, non-integer generations,
and signed improvement values.
"""

from __future__ import annotations

import pytest

from precision_health_os.evaluation import EvolutionTracker


def make_tracker(metric: str = "score") -> EvolutionTracker:
    return EvolutionTracker(metric=metric)


class TestEmptyTracker:
    """Every method must be safe on a freshly constructed tracker."""

    def test_best_generation_is_none(self) -> None:
        assert make_tracker().best_generation() is None

    def test_best_score_is_zero(self) -> None:
        assert make_tracker().best_score() == 0.0

    def test_improvement_is_zero(self) -> None:
        assert make_tracker().improvement() == 0.0

    def test_is_improving_is_false(self) -> None:
        assert make_tracker().is_improving() is False

    def test_regressed_is_false(self) -> None:
        assert make_tracker().regressed() is False

    def test_history_is_empty(self) -> None:
        assert make_tracker().history() == []


class TestSingleEntry:
    """A single recorded generation is a valid but non-improving state."""

    def test_best_generation_matches(self) -> None:
        t = make_tracker()
        t.record(1, 0.42)
        assert t.best_generation() == 1

    def test_best_score_matches(self) -> None:
        t = make_tracker()
        t.record(1, 0.42)
        assert t.best_score() == 0.42

    def test_improvement_is_zero(self) -> None:
        t = make_tracker()
        t.record(1, 0.42)
        assert t.improvement() == 0.0

    def test_is_improving_is_false(self) -> None:
        t = make_tracker()
        t.record(1, 0.42)
        assert t.is_improving() is False

    def test_regressed_is_false(self) -> None:
        t = make_tracker()
        t.record(1, 0.42)
        assert t.regressed() is False


class TestMonotonicImprovement:
    """Strictly increasing scores are improving and never regressed."""

    def test_is_improving_true(self) -> None:
        t = make_tracker()
        for gen, score in enumerate([0.30, 0.50, 0.70, 0.95], start=1):
            t.record(gen, score)
        assert t.is_improving() is True

    def test_regressed_is_false(self) -> None:
        t = make_tracker()
        for gen, score in enumerate([0.30, 0.50, 0.70, 0.95], start=1):
            t.record(gen, score)
        assert t.regressed() is False

    def test_best_is_last(self) -> None:
        t = make_tracker()
        for gen, score in enumerate([0.30, 0.50, 0.70, 0.95], start=1):
            t.record(gen, score)
        assert t.best_generation() == 4
        assert t.best_score() == 0.95


class TestRegressionDetection:
    """A drop below the best score is a regression."""

    def test_regressed_true_after_drop(self) -> None:
        t = make_tracker()
        t.record(1, 0.90)
        t.record(2, 0.60)
        assert t.regressed() is True

    def test_best_score_still_the_peak(self) -> None:
        t = make_tracker()
        t.record(1, 0.90)
        t.record(2, 0.60)
        assert t.best_score() == 0.90

    def test_is_improving_false_after_drop(self) -> None:
        t = make_tracker()
        t.record(1, 0.90)
        t.record(2, 0.60)
        assert t.is_improving() is False

    def test_regressed_true_mid_sequence(self) -> None:
        t = make_tracker()
        for gen, score in enumerate([0.50, 0.80, 0.65], start=1):
            t.record(gen, score)
        assert t.regressed() is True
        assert t.best_score() == 0.80


class TestPlateau:
    """Equal consecutive scores are neither improving nor regressed."""

    def test_is_improving_false_on_plateau(self) -> None:
        t = make_tracker()
        for gen, score in enumerate([0.50, 0.50, 0.50], start=1):
            t.record(gen, score)
        assert t.is_improving() is False

    def test_regressed_false_on_plateau(self) -> None:
        t = make_tracker()
        for gen, score in enumerate([0.50, 0.50, 0.50], start=1):
            t.record(gen, score)
        assert t.regressed() is False

    def test_plateau_after_improvement(self) -> None:
        t = make_tracker()
        for gen, score in enumerate([0.40, 0.60, 0.60], start=1):
            t.record(gen, score)
        assert t.is_improving() is False
        assert t.regressed() is False


class TestRecoveryAfterRegression:
    """A new high after a dip clears the regression flag."""

    def test_recovery_clears_regression(self) -> None:
        t = make_tracker()
        for gen, score in enumerate([0.50, 0.30, 0.80], start=1):
            t.record(gen, score)
        assert t.regressed() is False

    def test_recovery_sets_new_best(self) -> None:
        t = make_tracker()
        for gen, score in enumerate([0.50, 0.30, 0.80], start=1):
            t.record(gen, score)
        assert t.best_score() == 0.80
        assert t.best_generation() == 3

    def test_recovery_is_not_monotonic(self) -> None:
        t = make_tracker()
        for gen, score in enumerate([0.50, 0.30, 0.80], start=1):
            t.record(gen, score)
        assert t.is_improving() is False


class TestHistoryImmutability:
    """Mutating the returned history must not affect the tracker."""

    def test_append_to_returned_list_is_isolated(self) -> None:
        t = make_tracker()
        t.record(1, 0.5)
        h = t.history()
        h.append({"generation": 99, "score": 9.9})
        assert len(t.history()) == 1

    def test_clear_returned_list_is_isolated(self) -> None:
        t = make_tracker()
        t.record(1, 0.5)
        t.record(2, 0.7)
        h = t.history()
        h.clear()
        assert len(t.history()) == 2

    def test_mutate_entry_in_returned_list_is_isolated(self) -> None:
        """Mutating a returned entry must not corrupt the tracker."""
        t = make_tracker()
        t.record(1, 0.5)
        h = t.history()
        h[0]["score"] = 99.0
        assert t.history()[0]["score"] == 0.5


class TestNonIntegerGenerations:
    """Generation labels need not be integers."""

    def test_float_generations(self) -> None:
        t = make_tracker()
        t.record(0.5, 0.30)
        t.record(1.5, 0.60)
        assert t.best_generation() == 1.5
        assert t.is_improving() is True

    def test_string_generations(self) -> None:
        t = make_tracker()
        t.record("gen-a", 0.40)
        t.record("gen-b", 0.80)
        assert t.best_generation() == "gen-b"
        assert t.improvement() == pytest.approx(0.40)


class TestImprovementSign:
    """improvement() is positive for gain, negative for loss."""

    def test_positive_for_gain(self) -> None:
        t = make_tracker()
        t.record(1, 0.40)
        t.record(2, 0.80)
        assert t.improvement() == pytest.approx(0.40)

    def test_negative_for_loss(self) -> None:
        t = make_tracker()
        t.record(1, 0.80)
        t.record(2, 0.40)
        assert t.improvement() == pytest.approx(-0.40)

    def test_zero_for_no_change(self) -> None:
        t = make_tracker()
        t.record(1, 0.50)
        t.record(2, 0.50)
        assert t.improvement() == 0.0

    def test_uses_first_and_last_not_adjacent(self) -> None:
        t = make_tracker()
        for gen, score in enumerate([0.20, 0.90, 0.30], start=1):
            t.record(gen, score)
        assert t.improvement() == pytest.approx(0.10)
