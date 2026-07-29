"""Tests for WikiSQL parsing and ground-truth SQL rendering."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_text_to_sql.wikisql import (  # noqa: E402
    WikiSQLExample,
    WikiSQLFormatError,
    WikiSQLTable,
    as_number,
    quote_identifier,
    quote_literal,
    render_sql,
)


def make_table(**overrides):
    payload = {
        "id": "1-10015132-11",
        "header": ["Player", "No.", "School/Club Team", "Points"],
        "types": ["text", "text", "text", "real"],
        "rows": [
            ["Antonio Lang", "21", "Duke", 12],
            ["Voshon Lenard", "2", "Minnesota", 9],
            ["Martin Lewis", "32, 44", "Butler CC (KS)", 10],
        ],
    }
    payload.update(overrides)
    return WikiSQLTable.from_json(payload)


def make_example(sel=0, agg=0, conds=None):
    return WikiSQLExample.from_json(
        {
            "table_id": "1-10015132-11",
            "question": "who?",
            "sql": {"sel": sel, "agg": agg, "conds": conds or []},
        }
    )


# --------------------------------------------------------------------------
# Identifier and literal quoting
# --------------------------------------------------------------------------


class TestQuoting:
    def test_plain_identifier_is_quoted(self):
        assert quote_identifier("Player") == '"Player"'

    @pytest.mark.parametrize(
        "header", ["School/Club Team", "Pick #", "No. in series", "Time/Retired"]
    )
    def test_headers_with_special_characters(self, header):
        """1,116 distinct headers in the corpus need this; unquoted they are
        syntax errors."""
        assert quote_identifier(header) == f'"{header}"'

    def test_embedded_double_quote_is_escaped(self):
        assert quote_identifier('a"b') == '"a""b"'

    def test_embedded_single_quote_is_escaped(self):
        """Values like "Bishop's Stortford" appear in the corpus and would
        otherwise terminate the literal early."""
        assert quote_literal("Bishop's Stortford") == "'Bishop''s Stortford'"


# --------------------------------------------------------------------------
# Numeric coercion
# --------------------------------------------------------------------------


class TestAsNumber:
    @pytest.mark.parametrize(
        "value,expected",
        [("12", 12.0), (12, 12.0), (12.5, 12.5), ("48,065", 48065.0), ("  7 ", 7.0)],
    )
    def test_numeric_values(self, value, expected):
        assert as_number(value) == expected

    @pytest.mark.parametrize("value", ["1999-2000", "32, 44", "", None, "Duke"])
    def test_non_numeric_returns_none(self, value):
        assert as_number(value) is None

    def test_booleans_are_not_numbers(self):
        """bool is a subclass of int; treating True as 1.0 would be wrong."""
        assert as_number(True) is None


# --------------------------------------------------------------------------
# Table naming
# --------------------------------------------------------------------------


class TestTableName:
    def test_explicit_name_is_preferred(self):
        assert make_table(name="table_10015132_11").name == "table_10015132_11"

    def test_name_derived_from_id_when_absent(self):
        """~75% of tables have no `name`; the derivation must match the
        convention used by the upstream prebuilt .db files."""
        assert make_table().name == "table_1_10015132_11"


# --------------------------------------------------------------------------
# SQL rendering
# --------------------------------------------------------------------------


class TestRenderSQL:
    def test_simple_projection(self):
        sql = render_sql(make_table(), make_example(sel=0))
        assert sql == 'SELECT "Player" FROM "table_1_10015132_11"'

    def test_aggregation_wraps_projection(self):
        sql = render_sql(make_table(), make_example(sel=3, agg=1))
        assert sql.startswith('SELECT MAX("Points")')

    def test_text_condition_is_quoted(self):
        sql = render_sql(make_table(), make_example(conds=[[2, 0, "Butler CC (KS)"]]))
        assert sql.endswith("WHERE \"School/Club Team\" = 'Butler CC (KS)'")

    def test_numeric_condition_is_unquoted(self):
        """Against a REAL column the literal must be numeric; quoted, SQLite
        would compare lexicographically and '9' > '10' would be true."""
        sql = render_sql(make_table(), make_example(conds=[[3, 1, "10"]]))
        assert sql.endswith('WHERE "Points" > 10')

    def test_integral_float_has_no_trailing_zero(self):
        sql = render_sql(make_table(), make_example(conds=[[3, 0, 10.0]]))
        assert sql.endswith("= 10")

    def test_non_numeric_value_on_real_column_falls_back_to_literal(self):
        sql = render_sql(make_table(), make_example(conds=[[3, 0, "n/a"]]))
        assert sql.endswith("= 'n/a'")

    def test_multiple_conditions_joined_with_and(self):
        sql = render_sql(
            make_table(), make_example(conds=[[0, 0, "Antonio Lang"], [3, 2, "20"]])
        )
        assert ' AND ' in sql and sql.count("WHERE") == 1

    def test_unused_operator_index_is_rejected(self):
        """Operator index 3 ('OP') is a placeholder that never occurs in the
        corpus. Rendering it as something invented would be worse than failing."""
        with pytest.raises(WikiSQLFormatError, match="placeholder"):
            render_sql(make_table(), make_example(conds=[[0, 3, "x"]]))

    def test_out_of_range_select_index_is_rejected(self):
        with pytest.raises(WikiSQLFormatError, match="select index"):
            render_sql(make_table(), make_example(sel=99))

    def test_out_of_range_condition_column_is_rejected(self):
        with pytest.raises(WikiSQLFormatError, match="condition column"):
            render_sql(make_table(), make_example(conds=[[99, 0, "x"]]))
