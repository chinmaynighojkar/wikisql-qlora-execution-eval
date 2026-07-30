"""Execution-match evaluation.

The premise of the whole project: two syntactically different SQL queries can
be semantically identical, so correctness is decided by *running* the query
and comparing result sets, not by comparing strings.

    gold:      SELECT "Opponent" FROM "t" WHERE "Game" = 33
    predicted: select Opponent from t where Game=33
    string match: FAIL        execution match: PASS

This module is used unchanged by Phase 2 (baseline) and Phase 4 (fine-tuned).
Only the model pointer differs between those runs -- see `scripts/run_eval.py`.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .extraction import ExtractionError, extract_sql, normalize_for_exact_match

# A model can emit a query that is valid but pathological (an unbounded
# cross join). Without a ceiling, one bad generation stalls the whole run.
DEFAULT_TIMEOUT_SECONDS = 5.0

# Tolerance for float comparison. SUM/AVG over REAL columns accumulate
# representation error, so exact equality would fail on arithmetically
# identical results.
FLOAT_TOLERANCE = 1e-6


class QueryTimeout(RuntimeError):
    pass


class UnknownIdentifier(RuntimeError):
    """A double-quoted identifier that does not name a real column."""


# --------------------------------------------------------------------------
# SQLite's double-quoted-string misfeature
# --------------------------------------------------------------------------


def _scan_double_quoted(statement: str) -> list[tuple[str, int]]:
    """Return every double-quoted token with its opening-quote offset,
    ignoring those inside single-quoted string literals."""
    tokens: list[tuple[str, int]] = []
    index = 0
    length = len(statement)
    while index < length:
        char = statement[index]
        if char == "'":  # skip a single-quoted literal wholesale
            index += 1
            while index < length:
                if statement[index] == "'":
                    if index + 1 < length and statement[index + 1] == "'":
                        index += 2
                        continue
                    break
                index += 1
            index += 1
        elif char == '"':
            opened_at = index
            index += 1
            start = index
            buffer: list[str] = []
            while index < length:
                if statement[index] == '"':
                    if index + 1 < length and statement[index + 1] == '"':
                        buffer.append('"')
                        index += 2
                        continue
                    break
                buffer.append(statement[index])
                index += 1
            tokens.append(("".join(buffer) if buffer else statement[start:index], opened_at))
            index += 1
        else:
            index += 1
    return tokens


# A double-quoted token sitting immediately after a comparison operator, or
# after LIKE / IN, is being used as a *value*, not as a column reference.
_VALUE_POSITION = re.compile(r"(?:[=<>!]|\bLIKE\b|\bIN\b\s*\(?|,)\s*$", re.IGNORECASE)


def _is_value_position(statement: str, offset: int) -> bool:
    return bool(_VALUE_POSITION.search(statement[:offset]))


def find_unknown_identifiers(statement: str, known: Sequence[str]) -> list[str]:
    """Find double-quoted identifiers that name nothing in the schema.

    This exists because of a genuine SQLite misfeature. For MySQL
    compatibility, SQLite silently reinterprets a double-quoted identifier as
    a *string literal* when it does not match a column:

        SELECT "Nope" FROM t   ->  [('Nope',), ('Nope',)]     no error
        SELECT [Nope] FROM t   ->  OperationalError: no such column
        SELECT  Nope  FROM t   ->  OperationalError: no such column

    Left alone this corrupts the evaluation. The prompt presents columns
    double-quoted, so a model hallucinating a column name produces exactly the
    form that fails silently: the query "succeeds", inflating the syntactic
    validity rate with queries that are in fact wrong.

    SQLite can disable this via `SQLITE_DBCONFIG_DQS_DML`, but Python only
    exposes `Connection.setconfig` from 3.12 and this project targets 3.10
    (D-006), so the check is done here instead.

    Two things are deliberately **not** flagged, because rejecting them would
    penalise queries that are actually correct:

    - **Aliases.** `SELECT "x" AS "label"` does not change the result set.
    - **Values.** A double-quoted token in a comparison position is a
      misquoted *string literal*, not a column reference. The Phase 0 baseline
      generation produced exactly this:

          SELECT School_Club_Team FROM t WHERE No. = "21"

      That is non-standard SQL -- `'21'` is correct -- but SQLite executes it
      and returns the right rows. Flagging it would penalise the un-fine-tuned
      baseline for a quoting habit, while the fine-tuned model learns `'...'`
      from the gold SQL, inflating the measured improvement with a scoring
      artefact rather than real learning. Since the query still has to return
      the correct result set to score, letting it through costs nothing: a
      hallucinated *column* in a value position would return the literal
      string and fail the execution-match comparison anyway.
    """
    known_folded = {k.casefold() for k in known}
    unknown: list[str] = []
    for token, offset in _scan_double_quoted(statement):
        if token.casefold() in known_folded:
            continue
        if _is_value_position(statement, offset):
            continue
        # Allow aliases: "... AS "label"".
        pattern = re.compile(r"\bAS\s+\"" + re.escape(token) + r"\"", re.IGNORECASE)
        if pattern.search(statement):
            continue
        unknown.append(token)
    return unknown


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def execute_guarded(
    connection: sqlite3.Connection,
    statement: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[tuple[Any, ...]]:
    """Execute a statement with a wall-clock ceiling.

    SQLite has no query timeout, so a progress handler is installed: SQLite
    invokes it every N virtual-machine instructions, and a non-zero return
    aborts the query.
    """
    deadline = time.perf_counter() + timeout_seconds

    def _interrupt() -> int:
        return 1 if time.perf_counter() > deadline else 0

    connection.set_progress_handler(_interrupt, 10_000)
    try:
        return connection.execute(statement).fetchall()
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise QueryTimeout(f"exceeded {timeout_seconds}s") from exc
        raise
    finally:
        connection.set_progress_handler(None, 0)


# --------------------------------------------------------------------------
# Result comparison
# --------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    """True for real numbers. `bool` is excluded: it subclasses `int`, and
    treating `True` as `1.0` would silently equate a flag with a count."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_value(value: Any) -> Any:
    """Canonicalise one cell for comparison.

    Numbers are left as floats -- unifying int and float, so `COUNT`'s `12`
    and a REAL column's `12.0` are the same answer -- but deliberately *not*
    rounded here. Rounding into buckets is not a tolerance: two values closer
    together than the tolerance can straddle a bucket boundary and compare
    unequal. Numeric closeness is applied pairwise in `_values_equal`.

    Strings are casefolded and whitespace-collapsed. The official WikiSQL
    evaluation lowercases all values for the same reason; result casing is
    determined by the stored data rather than by the model, so this cannot let
    a wrong query pass.
    """
    if value is None:
        return None
    if _is_number(value):
        return float(value)
    return " ".join(str(value).split()).casefold()


def normalize_result(rows: Sequence[Sequence[Any]]) -> list[tuple[Any, ...]]:
    return [tuple(_normalize_value(v) for v in row) for row in rows]


def _values_equal(left: Any, right: Any) -> bool:
    """Compare two normalised cells, with a genuine numeric tolerance.

    `rel_tol` handles large magnitudes (WikiSQL contains values like
    339333.011497678, where an absolute 1e-6 would be punishingly strict) and
    `abs_tol` handles values near zero, where a relative tolerance is useless.
    """
    if left is None or right is None:
        return left is None and right is None
    left_num, right_num = _is_number(left), _is_number(right)
    if left_num != right_num:
        # A number and a string are different answers -- typically the model
        # selected a different column.
        return False
    if left_num:
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=FLOAT_TOLERANCE)
    return left == right


def _sort_key(row: Sequence[Any]) -> tuple:
    """Order rows deterministically for positional comparison.

    Numbers are bucketed *here only*, so values within the tolerance sort
    adjacently and line up for the pairwise check. The bucketing never decides
    equality -- `_values_equal` does.
    """
    key = []
    for value in row:
        if value is None:
            key.append((0, 0.0, ""))
        elif _is_number(value):
            key.append((1, round(value / FLOAT_TOLERANCE) * FLOAT_TOLERANCE, ""))
        else:
            key.append((2, 0.0, str(value)))
    return tuple(key)


def results_match(gold: Sequence[Sequence[Any]], predicted: Sequence[Sequence[Any]]) -> bool:
    """Compare result sets ignoring row order.

    Order-insensitive because none of the WikiSQL gold queries contain
    `ORDER BY`, so row order carries no meaning and SQLite does not guarantee
    it. Column order *is* significant: selecting different columns is a
    different answer, so tuples are compared positionally.

    Multiplicity is preserved (sorted lists, not sets) -- returning a row twice
    is not the same answer as returning it once.
    """
    left = sorted(normalize_result(gold), key=_sort_key)
    right = sorted(normalize_result(predicted), key=_sort_key)
    if len(left) != len(right):
        return False
    return all(
        len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
        for a, b in zip(left, right)
    )


# --------------------------------------------------------------------------
# Per-example scoring
# --------------------------------------------------------------------------


@dataclass
class ExampleResult:
    table_id: str
    question: str
    gold_sql: str
    raw_output: str
    predicted_sql: str | None = None
    executed: bool = False
    execution_match: bool = False
    exact_match: bool = False
    failure_reason: str | None = None
    gold_rows: list[Any] = field(default_factory=list)
    predicted_rows: list[Any] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "question": self.question,
            "gold_sql": self.gold_sql,
            "predicted_sql": self.predicted_sql,
            "raw_output": self.raw_output,
            "executed": self.executed,
            "execution_match": self.execution_match,
            "exact_match": self.exact_match,
            "failure_reason": self.failure_reason,
        }


def score_example(
    connection: sqlite3.Connection,
    record: dict[str, Any],
    raw_output: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ExampleResult:
    """Score one generation against its gold query."""
    result = ExampleResult(
        table_id=record["table_id"],
        question=record["question"],
        gold_sql=record["gold_sql"],
        raw_output=raw_output,
    )

    try:
        result.predicted_sql = extract_sql(raw_output)
    except ExtractionError as exc:
        result.failure_reason = f"extraction: {exc}"
        return result

    result.exact_match = normalize_for_exact_match(
        result.predicted_sql
    ) == normalize_for_exact_match(record["gold_sql"])

    # Guard against SQLite's double-quoted-string misfeature before executing;
    # see `find_unknown_identifiers`. A hallucinated column would otherwise
    # execute cleanly as a string literal and be counted as valid SQL.
    known = list(record["columns"]) + [record["table_name"]]
    unknown = find_unknown_identifiers(result.predicted_sql, known)
    if unknown:
        result.failure_reason = "unknown_identifier: " + ", ".join(sorted(set(unknown))[:3])
        return result

    try:
        predicted_rows = execute_guarded(connection, result.predicted_sql, timeout_seconds)
    except QueryTimeout as exc:
        result.failure_reason = f"timeout: {exc}"
        return result
    except sqlite3.Error as exc:
        result.failure_reason = f"sql_error: {type(exc).__name__}: {exc}"
        return result

    result.executed = True

    # The gold query is re-executed here rather than trusted from Phase 1, so
    # both sides of the comparison come from the same connection and the same
    # code path on every run.
    gold_rows = execute_guarded(connection, record["gold_sql"], timeout_seconds)

    result.gold_rows = [list(r) for r in gold_rows[:5]]
    result.predicted_rows = [list(r) for r in predicted_rows[:5]]
    result.execution_match = results_match(gold_rows, predicted_rows)
    if not result.execution_match:
        result.failure_reason = "wrong_result"
    return result


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate_metrics(results: Sequence[ExampleResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        raise ValueError("no results to aggregate")

    executed = sum(r.executed for r in results)
    matched = sum(r.execution_match for r in results)
    exact = sum(r.exact_match for r in results)
    extracted = sum(r.predicted_sql is not None for r in results)

    reasons: dict[str, int] = {}
    for r in results:
        if r.failure_reason:
            reasons[r.failure_reason.split(":")[0]] = (
                reasons.get(r.failure_reason.split(":")[0], 0) + 1
            )

    return {
        "n": total,
        # Primary metric.
        "execution_accuracy": round(matched / total, 4),
        # Secondary: did the query run at all?
        "syntactic_validity_rate": round(executed / total, 4),
        # Secondary: the brittle metric this project argues against relying on.
        "exact_match_accuracy": round(exact / total, 4),
        "sql_extraction_rate": round(extracted / total, 4),
        "counts": {
            "execution_match": matched,
            "executed": executed,
            "exact_match": exact,
            "sql_extracted": extracted,
        },
        "failure_breakdown": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def breakdown_by(
    results: Sequence[ExampleResult], records: Sequence[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    """Execution accuracy sliced by a record field (e.g. condition_count).

    A single headline number can hide that all the gain came from the easiest
    query shape, so the slice is reported alongside it.
    """
    buckets: dict[str, list[ExampleResult]] = {}
    for result, record in zip(results, records):
        buckets.setdefault(str(record.get(key)), []).append(result)
    return {
        bucket: {
            "n": len(group),
            "execution_accuracy": round(
                sum(r.execution_match for r in group) / len(group), 4
            ),
        }
        for bucket, group in sorted(buckets.items())
    }
