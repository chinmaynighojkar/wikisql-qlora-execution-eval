"""Tests for recovering SQL from raw model output.

Extraction is fairness-critical. The baseline model is chattier than a
fine-tuned one, so extraction that is too strict penalises the baseline and
manufactures an improvement fine-tuning did not earn; extraction that is too
loose invents queries the model did not produce. Both directions are tested.
"""

import pytest

from lora_text_to_sql.extraction import (
    ExtractionError,
    extract_sql,
    normalize_for_exact_match,
)

GOLD = 'SELECT "Opponent" FROM "t" WHERE "Game" = 33'


class TestCleanOutput:
    def test_bare_statement_passes_through(self):
        assert extract_sql(GOLD) == GOLD

    def test_trailing_semicolon_removed(self):
        assert extract_sql(GOLD + ";") == GOLD

    def test_surrounding_whitespace_collapsed(self):
        assert extract_sql(f"\n\n  {GOLD}  \n") == GOLD


class TestMarkdownAndPreamble:
    def test_sql_fence(self):
        assert extract_sql(f"```sql\n{GOLD}\n```") == GOLD

    def test_bare_fence(self):
        assert extract_sql(f"```\n{GOLD}\n```") == GOLD

    def test_unterminated_fence(self):
        """Truncation at max_new_tokens routinely leaves the closing fence off."""
        assert extract_sql(f"```sql\n{GOLD}\n") == GOLD

    def test_sql_prefix(self):
        assert extract_sql(f"SQL: {GOLD}") == GOLD

    def test_conversational_preamble(self):
        assert extract_sql(f"Sure! Here is the query:\n\n{GOLD}") == GOLD

    def test_preamble_and_fence_and_trailing_prose(self):
        raw = f"Here you go:\n\n```sql\n{GOLD};\n```\n\nThis returns the opponent."
        assert extract_sql(raw) == GOLD


class TestTrailingContent:
    def test_prose_after_semicolon_dropped(self):
        assert extract_sql(f"{GOLD}; This query finds the opponent.") == GOLD

    def test_prose_after_blank_line_dropped(self):
        assert extract_sql(f"{GOLD}\n\nThis query finds the opponent.") == GOLD

    def test_second_statement_dropped(self):
        assert extract_sql(f'{GOLD}; SELECT "Other" FROM "t"') == GOLD

    def test_semicolon_inside_literal_is_not_a_terminator(self):
        """Splitting naively on ';' would truncate this valid query."""
        statement = "SELECT \"A\" FROM \"t\" WHERE \"B\" = 'x;y'"
        assert extract_sql(statement) == statement

    def test_multiline_query_preserved(self):
        raw = 'SELECT "Opponent"\nFROM "t"\nWHERE "Game" = 33'
        assert extract_sql(raw) == GOLD


class TestRejection:
    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_empty_output_rejected(self, raw):
        with pytest.raises(ExtractionError):
            extract_sql(raw)

    def test_non_sql_prose_rejected(self):
        with pytest.raises(ExtractionError, match="no SELECT"):
            extract_sql("I am not able to answer that question.")

    @pytest.mark.parametrize(
        "raw",
        [
            'SELECT "A" FROM "t"; DROP TABLE "t"',
            'SELECT "A" FROM "t"; DELETE FROM "t"',
        ],
    )
    def test_trailing_destructive_statement_is_dropped_not_executed(self, raw):
        """The second statement is discarded by first-statement splitting, so
        the harmless SELECT survives and nothing destructive can execute."""
        assert extract_sql(raw) == 'SELECT "A" FROM "t"'

    def test_destructive_keyword_inside_the_statement_is_rejected(self):
        with pytest.raises(ExtractionError, match="non-read-only"):
            extract_sql('SELECT "A" FROM "t" WHERE "B" IN (DELETE FROM x)')


class TestNormalizeForExactMatch:
    def test_keyword_case_ignored(self):
        assert normalize_for_exact_match("select a from t") == normalize_for_exact_match(
            "SELECT A FROM T"
        )

    def test_whitespace_collapsed(self):
        assert normalize_for_exact_match("SELECT  a\n FROM t") == "select a from t"

    def test_quoting_style_is_not_absorbed(self):
        """Exact match is meant to be brittle -- if it absorbed quoting style
        it would stop being a useful contrast with execution match."""
        assert normalize_for_exact_match('SELECT "a" FROM t') != normalize_for_exact_match(
            "SELECT [a] FROM t"
        )
