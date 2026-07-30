"""Tests for execution-match scoring."""

import sqlite3

import pytest

from lora_text_to_sql.evaluate import (
    QueryTimeout,
    aggregate_metrics,
    execute_guarded,
    results_match,
    score_example,
)
from lora_text_to_sql.materialize import materialize_split
from lora_text_to_sql.wikisql import WikiSQLTable


@pytest.fixture
def connection(tmp_path):
    table = WikiSQLTable.from_json(
        {
            "id": "1-1-1",
            "name": "t",
            "header": ["Player", "Team", "Points"],
            "types": ["text", "text", "real"],
            "rows": [
                ["Antonio Lang", "Duke", "12"],
                ["Voshon Lenard", "Minnesota", "9"],
                ["Martin Lewis", "Butler CC (KS)", "10"],
            ],
        }
    )
    materialize_split([table], tmp_path / "t.db")
    return sqlite3.connect(tmp_path / "t.db")


RECORD = {
    "table_id": "1-1-1",
    "table_name": "t",
    "columns": ["Player", "Team", "Points"],
    "column_types": ["text", "text", "real"],
    "question": "Who plays for Duke?",
    "gold_sql": 'SELECT "Player" FROM "t" WHERE "Team" = \'Duke\'',
    "agg_index": 0,
    "condition_count": 1,
}


# --------------------------------------------------------------------------
# Result comparison
# --------------------------------------------------------------------------


class TestResultsMatch:
    def test_identical_results_match(self):
        assert results_match([("a",)], [("a",)])

    def test_row_order_is_ignored(self):
        """No gold query contains ORDER BY, so row order carries no meaning
        and SQLite does not guarantee it."""
        assert results_match([("a",), ("b",)], [("b",), ("a",)])

    def test_int_and_float_are_equivalent(self):
        """COUNT returns 12; a REAL column returns 12.0. Same answer."""
        assert results_match([(12,)], [(12.0,)])

    def test_float_tolerance(self):
        assert results_match([(1 / 3,)], [(0.3333333333333,)])

    def test_string_case_is_ignored(self):
        assert results_match([("Duke",)], [("duke",)])

    def test_different_values_do_not_match(self):
        assert not results_match([("a",)], [("b",)])

    def test_row_count_matters(self):
        assert not results_match([("a",)], [("a",), ("a",)])

    def test_duplicate_multiplicity_preserved(self):
        """Compared as sorted lists, not sets -- returning a row twice is not
        the same answer as returning it once."""
        assert not results_match([("a",), ("a",)], [("a",)])

    def test_column_order_matters(self):
        """Selecting the same columns in a different order is a different
        answer; tuples are compared positionally."""
        assert not results_match([("a", "b")], [("b", "a")])

    def test_empty_results_match_each_other(self):
        assert results_match([], [])


# --------------------------------------------------------------------------
# Guarded execution
# --------------------------------------------------------------------------


class TestExecuteGuarded:
    def test_normal_query_returns_rows(self, connection):
        assert execute_guarded(connection, 'SELECT "Player" FROM "t"') != []

    def test_runaway_query_times_out(self, connection):
        """A model can emit a valid but pathological query; without a ceiling
        one bad generation stalls the entire run."""
        runaway = (
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
            "SELECT COUNT(*) FROM c"
        )
        with pytest.raises(QueryTimeout):
            execute_guarded(connection, runaway, timeout_seconds=0.5)

    def test_invalid_sql_raises(self, connection):
        with pytest.raises(sqlite3.Error):
            execute_guarded(connection, "SELECT nope FROM nope")


# --------------------------------------------------------------------------
# Scoring one example
# --------------------------------------------------------------------------


class TestScoreExample:
    def test_gold_scores_a_match(self, connection):
        result = score_example(connection, RECORD, RECORD["gold_sql"])
        assert result.execution_match and result.executed and result.exact_match

    def test_semantically_equivalent_rewrite_matches(self, connection):
        """The project's whole premise: different text, same meaning."""
        result = score_example(connection, RECORD, "select Player from t where Team='duke'")
        assert result.execution_match
        assert not result.exact_match

    def test_wrapped_in_markdown_still_matches(self, connection):
        result = score_example(connection, RECORD, f"```sql\n{RECORD['gold_sql']};\n```")
        assert result.execution_match

    def test_wrong_column_does_not_match(self, connection):
        result = score_example(connection, RECORD, 'SELECT "Points" FROM "t" WHERE "Team" = \'Duke\'')
        assert result.executed
        assert not result.execution_match
        assert result.failure_reason == "wrong_result"

    def test_invalid_sql_is_recorded_as_sql_error(self, connection):
        result = score_example(connection, RECORD, "SELECT Nope FROM t")
        assert not result.executed
        assert result.failure_reason.startswith("sql_error")

    def test_hallucinated_double_quoted_column_is_not_counted_as_valid(self, connection):
        """SQLite silently turns an unknown double-quoted identifier into a
        string literal, so this query 'succeeds' and returns 'Nope' per row.
        Uncaught, it would inflate the syntactic validity rate with queries
        that are actually wrong -- and the prompt shows columns double-quoted,
        so it is the *likely* hallucination shape, not an exotic one."""
        result = score_example(connection, RECORD, 'SELECT "Nope" FROM "t"')
        assert not result.executed
        assert not result.execution_match
        assert result.failure_reason.startswith("unknown_identifier")

    def test_double_quoted_value_is_not_treated_as_a_column(self, connection):
        """Taken verbatim in shape from the Phase 0 baseline generation:
        `WHERE No. = "21"`. Non-standard SQL, but SQLite executes it and
        returns the right rows. Rejecting it would penalise the baseline for a
        quoting habit and inflate the fine-tuned model's apparent gain."""
        result = score_example(
            connection, RECORD, 'SELECT "Player" FROM "t" WHERE "Team" = "Duke"'
        )
        assert result.executed
        assert result.execution_match

    @pytest.mark.parametrize(
        "sql",
        [
            'SELECT "Player" FROM "t" WHERE "Team" > "Duke"',
            'SELECT "Player" FROM "t" WHERE "Team" LIKE "Duke"',
            'SELECT "Player" FROM "t" WHERE "Team" IN ("Duke")',
        ],
    )
    def test_value_positions_are_recognised(self, connection, sql):
        assert score_example(connection, RECORD, sql).executed

    def test_hallucinated_column_in_projection_is_still_caught(self, connection):
        """The value-position exemption must not weaken the original guard."""
        result = score_example(connection, RECORD, 'SELECT "Nope" FROM "t"')
        assert not result.executed
        assert result.failure_reason.startswith("unknown_identifier")

    def test_legitimate_alias_is_not_rejected(self, connection):
        result = score_example(
            connection, RECORD, 'SELECT "Player" AS "who" FROM "t" WHERE "Team" = \'Duke\''
        )
        assert result.executed and result.execution_match

    def test_non_sql_is_recorded_as_extraction_failure(self, connection):
        result = score_example(connection, RECORD, "I cannot answer that.")
        assert result.predicted_sql is None
        assert result.failure_reason.startswith("extraction")

    def test_query_returning_nothing_is_not_a_match(self, connection):
        """Guards the reason degenerate ground truth was excluded in Phase 1:
        if 'returns nothing' could match, valid nonsense would score."""
        result = score_example(
            connection, RECORD, 'SELECT "Player" FROM "t" WHERE "Team" = \'Nowhere\''
        )
        assert result.executed and not result.execution_match


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_metrics_on_mixed_results(self, connection):
        outputs = [
            RECORD["gold_sql"],                                   # match + exact
            "select Player from t where Team='duke'",             # match, not exact
            'SELECT "Points" FROM "t"',                           # runs, wrong
            "I cannot answer that.",                              # no SQL
        ]
        results = [score_example(connection, RECORD, o) for o in outputs]
        metrics = aggregate_metrics(results)

        assert metrics["n"] == 4
        assert metrics["execution_accuracy"] == 0.5
        assert metrics["syntactic_validity_rate"] == 0.75
        assert metrics["exact_match_accuracy"] == 0.25
        assert metrics["sql_extraction_rate"] == 0.75
        assert metrics["failure_breakdown"]["extraction"] == 1

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            aggregate_metrics([])
