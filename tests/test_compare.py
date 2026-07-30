"""Tests for the Phase 4 comparison statistics.

Written because mcnemar() and wilson_interval() compute the exact numbers
the README and resume bullets quote as proof the result is real (p < 1e-15,
non-overlapping confidence intervals), and neither function had a test
before this file: every other pure-logic function in the project does.

Writing these surfaced a real bug: mcnemar()'s discordant == 0 branch used a
different key schema ("fixed"/"broken"/"chi_square") than the normal return
("fixed_by_finetuning"/"chi_square_continuity_corrected", plus
discordant_pairs and p_value_display, which it lacked entirely).
render_markdown() reads the normal-path keys unconditionally, so a
comparison run with zero discordant pairs would have crashed with a
KeyError when rendering the report.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_phase4():
    spec = importlib.util.spec_from_file_location(
        "phase4_compare", REPO_ROOT / "scripts" / "phase4_compare.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase4_compare"] = module
    spec.loader.exec_module(module)
    return module


phase4 = load_phase4()
mcnemar = phase4.mcnemar
wilson_interval = phase4.wilson_interval


def _pairs(n_fixed, n_broken, n_both_correct, n_both_wrong, key="execution_match"):
    """Build (before, after) records with the requested counts of each outcome."""
    before, after = [], []
    for _ in range(n_fixed):
        before.append({key: False})
        after.append({key: True})
    for _ in range(n_broken):
        before.append({key: True})
        after.append({key: False})
    for _ in range(n_both_correct):
        before.append({key: True})
        after.append({key: True})
    for _ in range(n_both_wrong):
        before.append({key: False})
        after.append({key: False})
    return before, after


class TestMcNemarCounting:
    def test_classifies_all_four_outcomes_correctly(self):
        before, after = _pairs(n_fixed=10, n_broken=2, n_both_correct=5, n_both_wrong=3)
        result = mcnemar(before, after)
        assert result["fixed_by_finetuning"] == 10
        assert result["broken_by_finetuning"] == 2
        assert result["both_correct"] == 5
        assert result["both_wrong"] == 3
        assert result["discordant_pairs"] == 12

    def test_chi_square_matches_the_continuity_corrected_formula(self):
        before, after = _pairs(n_fixed=10, n_broken=2, n_both_correct=5, n_both_wrong=3)
        result = mcnemar(before, after)
        expected_chi_square = (abs(10 - 2) - 1) ** 2 / 12
        assert result["chi_square_continuity_corrected"] == round(expected_chi_square, 3)
        expected_p = math.erfc(math.sqrt(expected_chi_square) / math.sqrt(2))
        assert result["p_value"] == pytest.approx(expected_p, abs=1e-12)


class TestMcNemarRealData:
    def test_matches_the_committed_comparison_report(self):
        """Regression-locks the exact numbers reports/comparison.json quotes:
        238 fixed, 10 broken, chi-square 207.778, p < 1e-15."""
        before, after = _pairs(n_fixed=238, n_broken=10, n_both_correct=189, n_both_wrong=63)
        result = mcnemar(before, after)
        assert result["discordant_pairs"] == 248
        assert result["chi_square_continuity_corrected"] == pytest.approx(207.778, abs=0.001)
        assert result["p_value"] < 1e-15
        assert result["p_value_display"] == "<1e-15"
        assert result["significant_at_0_01"] is True


class TestMcNemarNoDiscordantPairs:
    def test_returns_the_same_schema_as_the_normal_path(self):
        """The regression this file exists to catch: a zero-discordant-pairs
        comparison must not crash render_markdown() by omitting keys it reads
        unconditionally."""
        before, after = _pairs(n_fixed=0, n_broken=0, n_both_correct=5, n_both_wrong=3)
        result = mcnemar(before, after)
        for key in (
            "fixed_by_finetuning", "broken_by_finetuning", "both_correct",
            "both_wrong", "discordant_pairs", "chi_square_continuity_corrected",
            "p_value", "p_value_display", "significant_at_0_01",
        ):
            assert key in result, f"missing {key!r} -- render_markdown() reads this unconditionally"
        assert result["discordant_pairs"] == 0
        assert result["chi_square_continuity_corrected"] == 0.0
        assert result["p_value"] == 1.0
        assert result["significant_at_0_01"] is False

    def test_empty_input_is_also_zero_discordant(self):
        result = mcnemar([], [])
        assert result["discordant_pairs"] == 0
        assert result["p_value"] == 1.0


class TestWilsonInterval:
    def test_zero_total_returns_zero_interval(self):
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_matches_the_committed_baseline_confidence_interval(self):
        """reports/comparison.json: baseline execution_match 199/500 ->
        95% Wilson interval [0.356, 0.4415]."""
        low, high = wilson_interval(199, 500)
        assert low == pytest.approx(0.356, abs=0.001)
        assert high == pytest.approx(0.4415, abs=0.001)

    def test_all_successes_stays_within_bounds(self):
        low, high = wilson_interval(500, 500)
        assert 0.0 <= low <= high <= 1.0

    def test_no_successes_stays_within_bounds(self):
        low, high = wilson_interval(0, 500)
        assert 0.0 <= low <= high <= 1.0

    def test_interval_widens_with_smaller_sample_size(self):
        """Same observed proportion, less data -- less certainty."""
        low_big, high_big = wilson_interval(50, 100)
        low_small, high_small = wilson_interval(5, 10)
        assert (high_small - low_small) > (high_big - low_big)
