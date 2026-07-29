"""Extract an executable SQL statement from raw model output.

An instruction-tuned model asked for "only SQL" will still routinely return
markdown fences, a `SQL:` prefix, a sentence of explanation, or several
statements. None of that is a modelling failure worth punishing -- the query
underneath may be perfectly correct -- so the harness extracts before it
scores.

This matters for fairness in both directions. The baseline model is chattier
than the fine-tuned one will be, so a naive harness that fed raw output
straight to SQLite would score the baseline near zero for formatting reasons
and manufacture an improvement that fine-tuning did not earn. Extraction is
applied identically to both runs.
"""

from __future__ import annotations

import re

# Statements that must never reach the database. The eval executes model
# output against a real file, so this is a hard guard rather than a nicety.
_FORBIDDEN = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)

_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)(?:```|$)", re.IGNORECASE | re.DOTALL)
_SELECT = re.compile(r"\bSELECT\b", re.IGNORECASE)
_LEADIN = re.compile(r"^\s*(?:SQL|Answer|Query|Output)\s*:\s*", re.IGNORECASE)


class ExtractionError(ValueError):
    """Raised when no usable SELECT statement can be recovered."""


def _split_first_statement(text: str) -> str:
    """Return text up to the first statement-terminating semicolon.

    Scans character by character rather than using `split(";")` because a
    semicolon inside a string literal (`WHERE "Name" = 'a;b'`) is data, not a
    terminator, and splitting on it would truncate a valid query.
    """
    in_single = False
    in_double = False
    for index, char in enumerate(text):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == ";" and not in_single and not in_double:
            return text[:index]
    return text


def _strip_trailing_prose(statement: str) -> str:
    """Drop explanatory text that follows the query when there was no semicolon.

    A blank line is the reliable boundary: models put prose in a new
    paragraph. Within a single block, a line that contains no SQL-ish token
    and reads like a sentence is also dropped.
    """
    block = statement.split("\n\n")[0]
    kept: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            break
        # A line starting with a capitalised word and ending in a full stop,
        # containing no SQL keyword, is prose.
        if kept and re.match(r"^[A-Z][a-z]", stripped) and stripped.endswith(".") and not re.search(
            r"\b(SELECT|FROM|WHERE|AND|OR|COUNT|SUM|AVG|MIN|MAX|GROUP|ORDER|LIMIT)\b",
            stripped,
            re.IGNORECASE,
        ):
            break
        kept.append(line)
    return "\n".join(kept)


def extract_sql(raw_output: str) -> str:
    """Recover a single executable SELECT statement from model output.

    Raises `ExtractionError` when nothing usable is present. That is a real
    outcome and is counted -- it is what "the model failed to produce SQL"
    looks like, and it is scored as a miss rather than hidden.
    """
    if raw_output is None:
        raise ExtractionError("no output")

    text = raw_output.strip()
    if not text:
        raise ExtractionError("empty output")

    # Prefer the contents of a fenced block when one is present.
    fence = _FENCE.search(text)
    if fence and fence.group(1).strip():
        text = fence.group(1).strip()

    text = _LEADIN.sub("", text).strip()

    match = _SELECT.search(text)
    if not match:
        raise ExtractionError("no SELECT statement found")
    text = text[match.start():]

    statement = _split_first_statement(text)
    statement = _strip_trailing_prose(statement)
    statement = " ".join(statement.split()).strip().rstrip(";").strip()

    if not statement:
        raise ExtractionError("empty statement after cleaning")
    if _FORBIDDEN.search(statement):
        raise ExtractionError("statement contains a non-read-only keyword")
    return statement


def normalize_for_exact_match(statement: str) -> str:
    """Casefold and collapse whitespace for the secondary exact-match metric.

    Exact match is reported only as a secondary signal. It is a deliberately
    strict, brittle measure -- it is the metric this project argues *against*
    relying on -- so it is normalised just enough that trivial whitespace and
    keyword-casing differences do not dominate it.
    """
    return " ".join(statement.split()).casefold().rstrip(";").strip()
