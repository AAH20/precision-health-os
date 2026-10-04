"""Regression tests for EWMA cold-start false positives.

Regression: detect_ewma() trusted its running variance from the very first
observation. With only one or two prior points the variance estimate is
unstable, so a normal reading (e.g. 72.55 bpm against a 70 bpm baseline)
scored 7.5 sigma and produced a false clinical alert.

Fix: EWMA abstains until it has enough prior observations to estimate
variance meaningfully, and the ensemble ignores abstaining votes.
"""

from __future__ import annotations

import random

from precision_health_os.iot import AnomalyDetector


class TestEWMAWarmup:
    """EWMA must not vote before its variance estimate is meaningful."""

    def test_early_points_are_not_scored(self) -> None:
        """The first few observations cannot produce an anomaly verdict."""
        detector = AnomalyDetector()
        # A 9-point series whose only 'excursion' is at index 2 — well inside
        # any reasonable warm-up window.
        series = [70.0, 70.5, 85.0, 70.0, 70.0, 70.5, 70.0, 70.0, 70.0]
        results = detector.detect_ewma(series)
        assert not results[2].is_anomaly, "flagged during EWMA warm-up"

    def test_warmup_flag_is_reported(self) -> None:
        """Warm-up abstentions must be visible, not silently dropped."""
        detector = AnomalyDetector()
        results = detector.detect_ewma([70.0, 70.5, 85.0, 70.0, 70.0, 70.5, 70.0])
        assert results[1].details.get("warmup") is True
        assert results[6].details.get("warmup") is False

    def test_late_spike_still_detected(self) -> None:
        """Warm-up must not cost recall on a spike at the end of the series."""
        detector = AnomalyDetector()
        for spike in (100, 150, 200, 500):
            series = [70, 72, 71, 70, 73, 71, 72, 71, spike]
            assert detector.detect_ewma(series)[-1].is_anomaly, f"missed {spike}"

    def test_ensemble_ignores_warmup_abstentions(self) -> None:
        """An early excursion must not be flagged by an abstaining method."""
        detector = AnomalyDetector()
        series = [70.0, 70.5, 85.0, 70.0, 70.0, 70.5, 70.0, 70.0, 70.0]
        results = detector.detect_ensemble(series)
        assert not results[2].is_anomaly, "ensemble counted a warm-up vote"

    def test_false_positive_rate_under_budget(self) -> None:
        """Stationary noise must not exceed the 5% false-positive budget."""
        detector = AnomalyDetector()
        flagged = 0
        trials = 40
        for seed in range(trials):
            random.seed(seed)
            series = [70 + random.uniform(-3, 3) for _ in range(9)]
            if any(r.is_anomaly for r in detector.detect_ensemble(series)):
                flagged += 1
        assert flagged / trials <= 0.05, f"{flagged}/{trials} false positives"

    def test_real_spikes_all_caught_after_fix(self) -> None:
        """Specificity fix must not reduce recall."""
        detector = AnomalyDetector()
        for spike in (100, 150, 200, 500):
            series = [70, 72, 71, 70, 73, 71, 72, 71, spike]
            assert detector.detect_ensemble(series)[-1].is_anomaly, f"missed {spike}"
