"""WikiSQL corpus types and ground-truth SQL rendering.

WikiSQL does not ship SQL strings. Each example carries a *structured*
annotation -- a selected column index, an aggregation index, and a list of
`(column_index, operator_index, value)` conditions -- and the SQL text has to
be rendered from it. That rendering is done here, once, so the training
targets and the ground truth executed by the eval harness are produced by
exactly the same code path and cannot drift apart.

Source: the official release, https://github.com/salesforce/WikiSQL (BSD-3),
`data.tar.bz2`. Official split sizes: train 56,355 / dev 8,421 / test 15,878.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Index -> SQL keyword, fixed by the WikiSQL annotation format.
AGG_OPS: tuple[str, ...] = ("", "MAX", "MIN", "COUNT", "SUM", "AVG")

# The official format defines a fourth operator, "OP", as an unused
# placeholder. Verified across all 80,654 examples in train+dev+test: operator
# index 3 never occurs. It is therefore rejected rather than guessed at.
COND_OPS: tuple[str, ...] = ("=", ">", "<")
_UNUSED_COND_OP_INDEX = 3


class WikiSQLFormatError(ValueError):
    """Raised when an example violates an assumption verified against the corpus."""


# --------------------------------------------------------------------------
# Identifier and literal rendering
# --------------------------------------------------------------------------


def quote_identifier(name: str) -> str:
    """Double-quote a SQL identifier, escaping embedded double quotes.

    Not optional for this corpus: 1,116 distinct column headers contain
    characters that are illegal in a bare identifier -- `Time/Retired`,
    `Pick #`, `No. in series`. Unquoted, those are syntax errors.
    """
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: Any) -> str:
    """Render a value as a single-quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def as_number(value: Any) -> float | None:
    """Best-effort numeric coercion. Returns None if the value is not numeric.

    Thousands separators are stripped: WikiSQL stores attendance figures and
    similar as `"48,065"`, which `float()` rejects outright.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Corpus types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WikiSQLTable:
    id: str
    header: list[str]
    types: list[str]
    rows: list[list[Any]]
    raw_name: str | None = None

    @property
    def name(self) -> str:
        """SQL table name.

        Only ~25% of tables carry an explicit `name` field, so it is derived
        from `id` for the rest. The derivation matches the convention used by
        the official prebuilt `.db` files (id `1-10015132-11` ->
        `table_1_10015132_11`), which keeps table names consistent with the
        upstream release.
        """
        if self.raw_name:
            return self.raw_name
        return "table_" + self.id.replace("-", "_")

    @property
    def column_count(self) -> int:
        return len(self.header)

    def is_numeric(self, column_index: int) -> bool:
        return self.types[column_index] == "real"

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> WikiSQLTable:
        return cls(
            id=payload["id"],
            header=list(payload["header"]),
            types=list(payload["types"]),
            rows=[list(r) for r in payload["rows"]],
            raw_name=payload.get("name"),
        )


@dataclass(frozen=True)
class WikiSQLExample:
    table_id: str
    question: str
    select_index: int
    agg_index: int
    conditions: list[tuple[int, int, Any]]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> WikiSQLExample:
        sql = payload["sql"]
        return cls(
            table_id=payload["table_id"],
            question=payload["question"].strip(),
            select_index=sql["sel"],
            agg_index=sql["agg"],
            conditions=[(c[0], c[1], c[2]) for c in sql["conds"]],
        )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_tables(path: Path) -> dict[str, WikiSQLTable]:
    tables: dict[str, WikiSQLTable] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            table = WikiSQLTable.from_json(json.loads(line))
            tables[table.id] = table
    return tables


def load_examples(path: Path) -> Iterator[WikiSQLExample]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield WikiSQLExample.from_json(json.loads(line))


# --------------------------------------------------------------------------
# Ground-truth SQL rendering
# --------------------------------------------------------------------------


def render_sql(table: WikiSQLTable, example: WikiSQLExample) -> str:
    """Render the structured annotation into an executable SQL string.

    Literals are inlined rather than parameterised on purpose. The model emits
    literal SQL text, so rendering the ground truth the same way means the
    reference and the prediction travel an identical execution path -- if
    literal quoting were broken, it would break for both and be caught here in
    Phase 1 rather than silently skewing Phase 2 and Phase 4.
    """
    if not 0 <= example.select_index < table.column_count:
        raise WikiSQLFormatError(
            f"select index {example.select_index} out of range for table {table.id}"
        )
    if not 0 <= example.agg_index < len(AGG_OPS):
        raise WikiSQLFormatError(f"unknown aggregation index {example.agg_index}")

    selected = quote_identifier(table.header[example.select_index])
    aggregation = AGG_OPS[example.agg_index]
    projection = f"{aggregation}({selected})" if aggregation else selected

    statement = f"SELECT {projection} FROM {quote_identifier(table.name)}"

    clauses: list[str] = []
    for column_index, operator_index, value in example.conditions:
        if operator_index == _UNUSED_COND_OP_INDEX:
            raise WikiSQLFormatError(
                "operator index 3 ('OP') is an unused placeholder in the WikiSQL "
                "format and does not occur anywhere in the corpus"
            )
        if not 0 <= operator_index < len(COND_OPS):
            raise WikiSQLFormatError(f"unknown operator index {operator_index}")
        if not 0 <= column_index < table.column_count:
            raise WikiSQLFormatError(
                f"condition column index {column_index} out of range for table {table.id}"
            )

        column = quote_identifier(table.header[column_index])
        operator = COND_OPS[operator_index]

        # Render numerically against REAL columns so comparisons are numeric
        # rather than lexicographic ('9' > '10' is true as text).
        if table.is_numeric(column_index):
            number = as_number(value)
            rendered = quote_literal(value) if number is None else _format_number(number)
        else:
            rendered = quote_literal(value)

        clauses.append(f"{column} {operator} {rendered}")

    if clauses:
        statement += " WHERE " + " AND ".join(clauses)
    return statement


def _format_number(number: float) -> str:
    """Render a float without a trailing `.0` when it is integral."""
    if number.is_integer():
        return str(int(number))
    return repr(number)
