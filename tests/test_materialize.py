"""Tests for SQLite materialisation.

The collation and numeric-typing tests here are regression guards on the two
decisions that most affect the project's headline numbers. If either is
quietly reverted, the ground-truth reference silently degrades and every
before/after metric moves for a reason that has nothing to do with
fine-tuning.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_text_to_sql.materialize import (  # noqa: E402
    create_table_statement,
    execute,
    has_colliding_headers,
    is_degenerate,
    materialize_split,
    validate_example,
)
from lora_text_to_sql.wikisql import WikiSQLExample, WikiSQLTable, render_sql  # noqa: E402


@pytest.fixture
def table():
    return WikiSQLTable.from_json(
        {
            "id": "1-1-1",
            "name": "t",
            "header": ["Player", "School/Club Team", "Points"],
            "types": ["text", "text", "real"],
            "rows": [
                ["Antonio Lang", "Duke", "12"],
                ["Voshon Lenard", "Minnesota", "9"],
                ["Martin Lewis", "Butler CC (KS)", "10"],
            ],
        }
    )


@pytest.fixture
def connection(tmp_path, table):
    materialize_split([table], tmp_path / "t.db")
    return sqlite3.connect(tmp_path / "t.db")


def example(sel=0, agg=0, conds=None):
    return WikiSQLExample.from_json(
        {"table_id": "1-1-1", "question": "q", "sql": {"sel": sel, "agg": agg, "conds": conds or []}}
    )


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


class TestSchema:
    def test_text_columns_declare_nocase_collation(self, table):
        statement = create_table_statement(table)
        assert '"Player" TEXT COLLATE NOCASE' in statement

    def test_real_columns_are_not_collated(self, table):
        assert '"Points" REAL' in statement_of(table)
        assert '"Points" REAL COLLATE' not in statement_of(table)


def statement_of(table):
    return create_table_statement(table)


# --------------------------------------------------------------------------
# The collation decision
# --------------------------------------------------------------------------


class TestCaseInsensitiveMatching:
    """WikiSQL condition values come from the question text, so their casing
    routinely differs from the stored cell. Measured on 500 dev examples:
    55.2% of ground-truth queries return a usable result with case-sensitive
    columns, 94.4% with COLLATE NOCASE."""

    @pytest.mark.parametrize("value", ["Duke", "duke", "DUKE", "dUkE"])
    def test_text_equality_ignores_case(self, connection, value):
        rows = execute(connection, f"SELECT \"Player\" FROM \"t\" WHERE \"School/Club Team\" = '{value}'")
        assert rows == [("Antonio Lang",)]

    def test_realistic_lowercased_condition_matches(self, connection):
        """The exact shape that fails without the collation."""
        rows = execute(
            connection,
            "SELECT \"Player\" FROM \"t\" WHERE \"School/Club Team\" = 'butler cc (ks)'",
        )
        assert rows == [("Martin Lewis",)]


# --------------------------------------------------------------------------
# The numeric-typing decision
# --------------------------------------------------------------------------


class TestNumericTyping:
    def test_comparison_is_numeric_not_lexicographic(self, connection):
        """Stored as text, '9' > '10' is true and every range query is wrong.
        Stored as REAL, only 12 and 10 exceed 9."""
        rows = execute(connection, 'SELECT "Player" FROM "t" WHERE "Points" > 9')
        assert sorted(r[0] for r in rows) == ["Antonio Lang", "Martin Lewis"]

    def test_aggregate_over_real_column(self, connection):
        assert execute(connection, 'SELECT SUM("Points") FROM "t"') == [(31.0,)]


# --------------------------------------------------------------------------
# Degenerate results
# --------------------------------------------------------------------------


class TestIsDegenerate:
    def test_empty_result_is_degenerate(self):
        assert is_degenerate([])

    def test_all_null_result_is_degenerate(self):
        """MAX() over zero matching rows returns (None,), not an empty set."""
        assert is_degenerate([(None,)])

    def test_real_value_is_not_degenerate(self):
        assert not is_degenerate([("Duke",)])

    def test_zero_is_not_degenerate(self):
        """COUNT() legitimately returns 0; that is information, not absence."""
        assert not is_degenerate([(0,)])

    def test_aggregate_with_no_matching_rows_is_flagged(self, connection, table):
        outcome = validate_example(connection, table, example(sel=2, agg=1, conds=[[0, 0, "Nobody"]]))
        assert outcome.status == "degenerate"

    def test_matching_query_is_usable(self, connection, table):
        outcome = validate_example(connection, table, example(sel=0, conds=[[1, 0, "duke"]]))
        assert outcome.status == "usable"
        assert outcome.result == [("Antonio Lang",)]


# --------------------------------------------------------------------------
# Header collisions
# --------------------------------------------------------------------------


class TestCollidingHeaders:
    def test_case_only_collision_is_detected(self):
        """SQLite compares column names case-insensitively, so these two
        headers cannot coexist. Four tables in the corpus hit this."""
        table = WikiSQLTable.from_json(
            {"id": "x", "header": ["% Total budget", "% total budget"], "types": ["text", "text"], "rows": []}
        )
        assert has_colliding_headers(table)

    def test_distinct_headers_do_not_collide(self, table):
        assert not has_colliding_headers(table)

    def test_colliding_tables_are_skipped_not_fatal(self, tmp_path):
        good = WikiSQLTable.from_json(
            {"id": "g", "name": "g", "header": ["A"], "types": ["text"], "rows": [["x"]]}
        )
        bad = WikiSQLTable.from_json(
            {"id": "b", "name": "b", "header": ["A", "a"], "types": ["text", "text"], "rows": []}
        )
        stats = materialize_split([good, bad], tmp_path / "m.db")
        assert stats.tables_written == 1
        assert stats.skipped_table_ids == ["b"]


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_rendered_ground_truth_executes(connection, table):
    """The rendering and execution paths must agree -- this is the whole basis
    of execution-match scoring."""
    sql = render_sql(table, example(sel=0, conds=[[1, 0, "Butler CC (KS)"]]))
    assert execute(connection, sql) == [("Martin Lewis",)]
