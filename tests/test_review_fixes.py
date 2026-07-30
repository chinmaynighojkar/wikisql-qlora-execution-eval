"""Regression tests for defects found in a second, independent code review.

Each test fails against the pre-review code. Grouped here rather than
scattered so the review's findings stay traceable to their fixes.

The zero-discordant-pairs key-schema fix these findings originally overlapped
with is already covered by tests/test_compare.py, so it is not repeated here.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from lora_text_to_sql.evaluate import FLOAT_TOLERANCE, results_match
from lora_text_to_sql.io import read_json, read_jsonl, read_only_connection

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


def outcomes(flags):
    return [{"execution_match": f} for f in flags]


# --------------------------------------------------------------------------
# Finding: Yates correction was not clamped at zero
# --------------------------------------------------------------------------


class TestYatesClamp:
    def test_tie_gives_zero_chi_square(self):
        """fixed == broken is no evidence of a difference. Unclamped,
        (|0| - 1)^2 = 1 produced a spurious chi-square of 1/discordant."""
        before = outcomes([True, False])
        after = outcomes([False, True])
        result = phase4.mcnemar(before, after)
        assert result["fixed_by_finetuning"] == result["broken_by_finetuning"] == 1
        assert result["chi_square_continuity_corrected"] == 0.0
        assert result["p_value"] == pytest.approx(1.0)

    def test_difference_of_one_also_clamps(self):
        before = outcomes([True, False, False])
        after = outcomes([False, True, True])
        result = phase4.mcnemar(before, after)
        assert result["chi_square_continuity_corrected"] == 0.0

    def test_large_difference_is_significant(self):
        before = outcomes([False] * 100 + [True] * 5)
        after = outcomes([True] * 100 + [False] * 5)
        result = phase4.mcnemar(before, after)
        assert result["chi_square_continuity_corrected"] > 50
        assert result["significant_at_0_01"] is True


# --------------------------------------------------------------------------
# Finding: float "tolerance" was bucketed rounding, not a tolerance
# --------------------------------------------------------------------------


class TestFloatTolerance:
    def test_values_inside_tolerance_match(self):
        """These straddled a rounding-bucket boundary and compared unequal
        despite being 2e-7 apart with a 1e-6 tolerance."""
        assert abs(1.0000004 - 1.0000006) < FLOAT_TOLERANCE
        assert results_match([(1.0000004,)], [(1.0000006,)])

    @pytest.mark.parametrize(
        "a,b",
        [
            (0.1 + 0.2, 0.3),            # classic representation error
            (1 / 3, 0.3333333333333333),
            (2.0000001, 2.0000002),
        ],
    )
    def test_arithmetically_equal_values_match(self, a, b):
        assert results_match([(a,)], [(b,)])

    def test_large_magnitude_uses_relative_tolerance(self):
        """WikiSQL contains values like 339333.011497678, where an absolute
        1e-6 tolerance would be punishingly strict."""
        assert results_match([(339333.011497678,)], [(339333.0114976781,)])

    def test_genuinely_different_values_still_fail(self):
        assert not results_match([(1.0,)], [(1.1,)])
        assert not results_match([(100.0,)], [(101.0,)])

    def test_int_float_equivalence_preserved(self):
        assert results_match([(12,)], [(12.0,)])

    def test_number_and_string_are_not_equal(self):
        """Typically means the model selected a different column."""
        assert not results_match([(12.0,)], [("12",)])

    def test_bool_is_not_a_number(self):
        assert not results_match([(True,)], [(1.0,)])

    def test_row_order_still_ignored_with_close_floats(self):
        assert results_match(
            [(1.0000004,), (5.0,)], [(5.0,), (1.0000006,)]
        )

    def test_multiplicity_still_preserved(self):
        assert not results_match([(1.0,), (1.0,)], [(1.0,)])

    def test_none_handling(self):
        assert results_match([(None,)], [(None,)])
        assert not results_match([(None,)], [(0.0,)])


# --------------------------------------------------------------------------
# Finding: shared IO helpers and connection lifetime
# --------------------------------------------------------------------------


class TestSharedIO:
    def test_read_jsonl_skips_blank_lines(self):
        """One of the three duplicated loaders skipped blanks and the others
        did not -- the exact kind of quiet divergence consolidation prevents."""
        path = Path(__file__).parent / "_tmp_blank.jsonl"
        path.write_text('{"a":1}\n\n{"a":2}\n\n', encoding="utf-8")
        try:
            assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
        finally:
            path.unlink()

    def test_read_jsonl_limit(self):
        path = Path(__file__).parent / "_tmp_limit.jsonl"
        path.write_text('{"a":1}\n{"a":2}\n{"a":3}\n', encoding="utf-8")
        try:
            assert len(read_jsonl(path, limit=2)) == 2
        finally:
            path.unlink()

    def test_missing_file_exits_with_guidance(self):
        with pytest.raises(SystemExit, match="phase1_prepare_data"):
            read_jsonl(Path("does_not_exist.jsonl"))

    def test_read_json_missing_file_exits(self):
        with pytest.raises(SystemExit):
            read_json(Path("does_not_exist.json"))

    def test_connection_is_closed_on_exit(self):
        import sqlite3

        from lora_text_to_sql.materialize import materialize_split
        from lora_text_to_sql.wikisql import WikiSQLTable

        tmp = Path(__file__).parent / "_tmp_io.db"
        table = WikiSQLTable.from_json(
            {"id": "1", "name": "t", "header": ["A"], "types": ["text"], "rows": [["x"]]}
        )
        materialize_split([table], tmp)
        try:
            with read_only_connection(tmp) as conn:
                assert conn.execute('SELECT "A" FROM "t"').fetchall() == [("x",)]
            with pytest.raises(sqlite3.ProgrammingError):
                conn.execute('SELECT "A" FROM "t"')  # closed
        finally:
            tmp.unlink(missing_ok=True)

    def test_connection_is_read_only(self):
        import sqlite3

        from lora_text_to_sql.materialize import materialize_split
        from lora_text_to_sql.wikisql import WikiSQLTable

        tmp = Path(__file__).parent / "_tmp_ro.db"
        table = WikiSQLTable.from_json(
            {"id": "1", "name": "t", "header": ["A"], "types": ["text"], "rows": [["x"]]}
        )
        materialize_split([table], tmp)
        try:
            with read_only_connection(tmp) as conn:
                with pytest.raises(sqlite3.OperationalError):
                    conn.execute('DELETE FROM "t"')
        finally:
            tmp.unlink(missing_ok=True)
