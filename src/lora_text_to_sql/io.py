"""Shared file and database helpers.

Consolidates three near-identical JSONL loaders that had drifted into
`dataset.py`, `run_eval.py` and `phase4_compare.py`. Small duplication, but
the kind that diverges quietly: one of the three silently skipped blank lines
and the others did not.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines.

    Raises `SystemExit` with a remediation hint rather than a bare
    `FileNotFoundError`, because the usual cause is a phase that has not been
    run yet.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run scripts/phase1_prepare_data.py first."
        )
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records[:limit] if limit else records


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the relevant phase first.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


@contextmanager
def read_only_connection(database_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite database read-only, and always close it.

    Read-only is the load-bearing part: these connections execute
    **model-generated** SQL. Extraction rejects non-read-only statements and
    only the first statement is taken, but `mode=ro` is the defence that does
    not depend on a regex being exhaustive -- SQLite itself refuses writes.
    """
    if not database_path.exists():
        raise SystemExit(
            f"{database_path} not found. Run scripts/phase1_prepare_data.py first."
        )
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        yield connection
    finally:
        connection.close()
