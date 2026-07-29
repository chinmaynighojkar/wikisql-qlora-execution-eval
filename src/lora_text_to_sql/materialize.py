"""Materialise WikiSQL tables into real SQLite databases.

This exists so generated SQL can be *executed* and compared by result set,
rather than compared as text. Two syntactically different queries can be
semantically identical, and string comparison scores those as wrong.

Why not use the `.db` files shipped in the official release: they anonymise
every column to `col0..colN` and lowercase every stored value. Prompting a
model with `col0, col1, col2` would reduce the task to guessing a column
index, which is not the task this project is about -- generating SQL against
real, human-readable schemas. So the tables are rebuilt here from
`*.tables.jsonl`, preserving the original headers and value casing.

Case handling is the single most consequential decision in this module; see
`_column_definition` for the measurement that drove it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .wikisql import (
    WikiSQLExample,
    WikiSQLTable,
    as_number,
    quote_identifier,
    render_sql,
)


class ExecutionError(RuntimeError):
    """Raised when a SQL statement fails to execute."""


# --------------------------------------------------------------------------
# Schema construction
# --------------------------------------------------------------------------


def _column_definition(header: str, type_name: str) -> str:
    """Build one column definition.

    Text columns are declared `COLLATE NOCASE`, which makes `=` comparisons on
    them case-insensitive. This is not a stylistic choice -- it was measured.
    On a random sample of 500 dev examples, executing the *ground-truth* SQL
    returned a usable result for:

        case-sensitive columns : 276 / 500  (55.2%)
        COLLATE NOCASE columns : 472 / 500  (94.4%)

    WikiSQL condition values are extracted from the question text, so their
    casing routinely differs from the stored cell ("russia" vs "Russia",
    "louisville" vs "Louisville"). Without NOCASE, roughly 45% of the
    reference queries return nothing, and every model would be scored against
    a reference that is broken almost half the time.

    The official WikiSQL evaluation solves this by lowercasing all stored
    values. Declaring the collation instead preserves the original casing for
    display in the prompt, while giving the same matching behaviour.

    Limitation, stated rather than hidden: SQLite's built-in NOCASE collation
    folds ASCII A-Z only. Case differences in non-ASCII text (e.g. "Cerro
    Porteño") are not folded.
    """
    declaration = "REAL" if type_name == "real" else "TEXT COLLATE NOCASE"
    return f"{quote_identifier(header)} {declaration}"


def create_table_statement(table: WikiSQLTable) -> str:
    columns = ", ".join(
        _column_definition(header, type_name)
        for header, type_name in zip(table.header, table.types)
    )
    return f"CREATE TABLE IF NOT EXISTS {quote_identifier(table.name)} ({columns})"


def has_colliding_headers(table: WikiSQLTable) -> bool:
    """True when two headers collide under SQLite's column-name comparison.

    SQLite compares column names case-insensitively, so a table with both
    `% Total budget` and `% total budget` is rejected at CREATE TABLE with
    "duplicate column name". The corpus contains no *exact* duplicate headers,
    but four tables (3 train, 1 test) collide only when case is folded.

    These tables are excluded rather than repaired. Repairing would mean
    renaming one column, which breaks the invariant that a column can be
    referenced by its real header name -- the annotation selects columns by
    index, so a renamed column would silently resolve to the wrong one.
    Exclusion is safe here because no example in any split references these
    four tables; the caller asserts that rather than assuming it.
    """
    lowered = [header.lower() for header in table.header]
    return len(set(lowered)) != len(lowered)


def _coerce_row(table: WikiSQLTable, row: Sequence[Any]) -> list[Any]:
    """Coerce a row to the declared column types.

    Values in `real` columns are converted to float so that `>` and `<`
    compare numerically. Stored as text, `'9' > '10'` is true, which would
    silently corrupt every comparison-based query.
    """
    coerced: list[Any] = []
    for index, value in enumerate(row):
        if table.is_numeric(index):
            coerced.append(as_number(value))
        else:
            coerced.append(None if value is None else str(value))
    return coerced


# --------------------------------------------------------------------------
# Execution helpers (shared with the Phase 2/4 eval harness)
# --------------------------------------------------------------------------


def execute(connection: sqlite3.Connection, statement: str) -> list[tuple[Any, ...]]:
    try:
        return connection.execute(statement).fetchall()
    except sqlite3.Error as exc:
        raise ExecutionError(f"{type(exc).__name__}: {exc}") from exc


def is_degenerate(result: list[tuple[Any, ...]]) -> bool:
    """True when a result carries no information worth scoring.

    Two cases, both of which must be excluded from the evaluation set:

    - An empty result set. A non-aggregate SELECT whose WHERE clause matched
      no rows.
    - An all-NULL result. `MAX(col)` over zero matching rows returns a single
      row containing NULL rather than an empty set.

    Both matter because the eval compares result sets. If the reference answer
    is "nothing", then *any* query that also returns nothing -- including
    syntactically valid nonsense selecting an unrelated column -- scores as
    correct. Keeping these examples would inflate both the baseline and the
    fine-tuned score by rewarding failure, and would do so unevenly.
    """
    if not result:
        return True
    return all(value is None for row in result for value in row)


# --------------------------------------------------------------------------
# Materialisation
# --------------------------------------------------------------------------


@dataclass
class MaterializationStats:
    split: str
    tables_written: int
    rows_written: int
    database_path: Path
    database_bytes: int
    skipped_table_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "tables_written": self.tables_written,
            "rows_written": self.rows_written,
            "tables_skipped_colliding_headers": len(self.skipped_table_ids),
            "skipped_table_ids": self.skipped_table_ids,
            "database_path": str(self.database_path),
            "database_mib": round(self.database_bytes / 1024**2, 2),
        }


def materialize_split(
    tables: Iterable[WikiSQLTable],
    database_path: Path,
    *,
    overwrite: bool = True,
) -> MaterializationStats:
    """Write every table of a split into one SQLite database.

    One database per split rather than one file per table: the train split has
    18,585 tables, and 18,585 separate files would be slow to create and
    awkward to move around. Table names are unique within a split, so a single
    file is unambiguous.
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and database_path.exists():
        database_path.unlink()

    tables_written = 0
    rows_written = 0
    skipped: list[str] = []

    connection = sqlite3.connect(database_path)
    try:
        # Durability is irrelevant here -- the file is a rebuildable artefact,
        # so the safety/speed trade-off is worth taking on 18.5K tables.
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")

        with connection:
            for table in tables:
                if has_colliding_headers(table):
                    skipped.append(table.id)
                    continue
                connection.execute(create_table_statement(table))
                placeholders = ", ".join("?" * table.column_count)
                connection.executemany(
                    f"INSERT INTO {quote_identifier(table.name)} VALUES ({placeholders})",
                    (_coerce_row(table, row) for row in table.rows),
                )
                tables_written += 1
                rows_written += len(table.rows)
    finally:
        connection.close()

    return MaterializationStats(
        split=database_path.stem,
        tables_written=tables_written,
        rows_written=rows_written,
        database_path=database_path,
        database_bytes=database_path.stat().st_size,
        skipped_table_ids=skipped,
    )


# --------------------------------------------------------------------------
# Ground-truth validation
# --------------------------------------------------------------------------


@dataclass
class ValidationOutcome:
    example: WikiSQLExample
    sql: str
    status: str  # "usable" | "degenerate" | "error"
    result: list[tuple[Any, ...]] | None = None
    error: str | None = None


def validate_example(
    connection: sqlite3.Connection,
    table: WikiSQLTable,
    example: WikiSQLExample,
) -> ValidationOutcome:
    """Render and execute one example's ground-truth SQL, and classify it."""
    try:
        statement = render_sql(table, example)
    except Exception as exc:  # noqa: BLE001 - surfaced as a data-quality status
        return ValidationOutcome(example, sql="", status="error", error=str(exc))

    try:
        result = execute(connection, statement)
    except ExecutionError as exc:
        return ValidationOutcome(example, sql=statement, status="error", error=str(exc))

    if is_degenerate(result):
        return ValidationOutcome(example, sql=statement, status="degenerate", result=result)
    return ValidationOutcome(example, sql=statement, status="usable", result=result)
