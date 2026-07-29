"""Tests for the Phase 3 loss-trend gate.

Written after the gate reported FAIL for a run that had trained correctly. A
200-record smoke run at `gradient_accumulation_steps=16` performs ~12
optimiser steps; at `logging_steps=10` that logs a single loss point, so
`losses[-1] < losses[0]` compared a number to itself and was always False.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_phase3():
    spec = importlib.util.spec_from_file_location(
        "phase3_train", REPO_ROOT / "scripts" / "phase3_train.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase3_train"] = module
    spec.loader.exec_module(module)
    return module


phase3 = load_phase3()
assess = phase3.assess_loss_trend


class TestInsufficientData:
    @pytest.mark.parametrize("losses", [[], [0.13], [0.13, 0.12], [0.13, 0.12, 0.11]])
    def test_too_few_points_is_indeterminate_not_failure(self, losses):
        """The exact regression: one logged point must not read as FAIL."""
        decreased, detail = assess(losses)
        assert decreased is None
        assert "logged loss point" in detail

    def test_the_real_smoke_run_is_indeterminate(self):
        """Reproduces the observed run: 200 records, 1 epoch, 1 logged point."""
        decreased, _ = assess([0.13232953548431398])
        assert decreased is None


class TestTrendDetection:
    def test_clearly_decreasing(self):
        decreased, detail = assess([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        assert decreased is True
        assert "-" in detail

    def test_clearly_increasing(self):
        assert assess([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[0] is False

    def test_flat_is_not_a_decrease(self):
        assert assess([0.5] * 8)[0] is False

    def test_noise_does_not_flip_the_verdict(self):
        """Endpoint comparison would call this an increase; the quartile means
        correctly see a downward trend."""
        losses = [1.0, 0.2, 0.95, 0.9, 0.5, 0.45, 0.3, 0.55]
        assert assess(losses)[0] is True

    def test_endpoint_comparison_would_have_been_wrong_here(self):
        losses = [1.0, 0.2, 0.95, 0.9, 0.5, 0.45, 0.3, 0.55]
        naive = losses[-1] < losses[0]
        assert naive is True and assess(losses)[0] is True

        # ...and the reverse: a final upward blip on a falling curve.
        losses = [1.0, 0.95, 0.9, 0.85, 0.4, 0.35, 0.3, 1.05]
        assert (losses[-1] < losses[0]) is False
        assert assess(losses)[0] is True

    def test_detail_reports_the_window_and_percentage(self):
        _, detail = assess([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5])
        assert "mean of first 2" in detail and "-50.0%" in detail


class TestMinPointsIsConfigurable:
    def test_lower_threshold_admits_shorter_runs(self):
        assert assess([0.5, 0.4], min_points=2)[0] is True
