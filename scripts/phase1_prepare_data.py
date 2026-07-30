"""Phase 1 gate: materialise WikiSQL into executable SQLite databases.

Produces, from the official WikiSQL release:

  data/tables/{train,dev,test}.db      real SQLite tables, one DB per split
  data/processed/train_subsample.jsonl stratified training subsample
  data/processed/test_eval.jsonl       the fixed evaluation subset
  reports/phase1_data_report.json      measured counts and validation results

Pass criteria, from the build plan: WikiSQL tables are materialised into
SQLite, and the ground-truth SQL executes cleanly on at least 20 samples.
This script checks that on a far larger sample and reports the real rate.

Usage:
    python scripts/phase1_prepare_data.py
    python scripts/phase1_prepare_data.py --validate-sample 2000 --eval-size 500
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sqlite3
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from lora_text_to_sql.materialize import (
    materialize_split,
    validate_example,
)
from lora_text_to_sql.wikisql import (
    WikiSQLExample,
    WikiSQLTable,
    load_examples,
    load_tables,
    render_sql,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The canonical release. This is the same archive the official Hugging Face
# loading script downloads. It is used directly because `datasets` >= 3 no
# longer executes dataset loading scripts, and every parquet mirror on the Hub
# is a preprocessed derivative that has dropped the table rows -- without
# which nothing can be executed. See docs/DECISIONS.md D-008.
DATA_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"

# Official split sizes, asserted after loading so a truncated or substituted
# download cannot pass silently.
EXPECTED_COUNTS = {"train": 56355, "dev": 8421, "test": 15878}

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TABLES_DIR = DATA_DIR / "tables"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_PATH = REPO_ROOT / "reports" / "phase1_data_report.json"

SEED = 20260728


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------


def _assert_safe_members(tar: tarfile.TarFile, destination: Path) -> None:
    """Reject any archive member that would extract outside `destination`.

    tarfile.extractall() has no built-in defence against a member path
    containing "../" or an absolute path (a CVE-2007-4559-class issue); the
    safe `filter="data"` option that closes this is Python 3.12+ only, and
    this project targets 3.10 (D-006). The archive comes from a fixed,
    trusted URL, so this is cheap defence-in-depth rather than a response to
    an observed problem: it costs nothing and means a compromised upstream
    (or a substituted URL) can't silently write files outside RAW_DIR.
    """
    destination = destination.resolve()
    for member in tar.getmembers():
        resolved = (destination / member.name).resolve()
        if resolved != destination and destination not in resolved.parents:
            raise SystemExit(f"refusing to extract {member.name!r}: escapes {destination}")
        if member.issym() or member.islnk():
            link_target = (resolved.parent / member.linkname).resolve()
            if link_target != destination and destination not in link_target.parents:
                raise SystemExit(
                    f"refusing to extract {member.name!r}: link target escapes {destination}"
                )


def ensure_raw_data(force: bool = False) -> Path:
    section("1. Acquire the official WikiSQL release")
    extracted = RAW_DIR / "data"
    marker = extracted / "train.jsonl"
    if marker.exists() and not force:
        print(f"  already present at {extracted}")
        return extracted

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "data.tar.bz2"
    if not archive.exists() or force:
        print(f"  downloading {DATA_URL}")
        started = time.perf_counter()
        urllib.request.urlretrieve(DATA_URL, archive)
        size_mib = archive.stat().st_size / 1024**2
        print(f"  downloaded {size_mib:.1f} MiB in {time.perf_counter() - started:.1f}s")

    print("  extracting")
    with tarfile.open(archive, "r:bz2") as tar:
        _assert_safe_members(tar, RAW_DIR)
        tar.extractall(RAW_DIR)
    return extracted


# --------------------------------------------------------------------------
# Load + materialise
# --------------------------------------------------------------------------


def load_split(
    raw_dir: Path, split: str
) -> tuple[dict[str, WikiSQLTable], list[WikiSQLExample]]:
    tables = load_tables(raw_dir / f"{split}.tables.jsonl")
    examples = list(load_examples(raw_dir / f"{split}.jsonl"))
    expected = EXPECTED_COUNTS[split]
    if len(examples) != expected:
        raise SystemExit(
            f"{split}: expected {expected} examples (official count), got "
            f"{len(examples)} -- the download looks truncated or substituted"
        )
    print(f"  {split:5}: {len(examples):6,} examples across {len(tables):6,} tables")
    return tables, examples


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_split(
    database_path: Path,
    tables: dict[str, WikiSQLTable],
    examples: list[WikiSQLExample],
    sample_size: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Execute the ground-truth SQL for a sample and classify every outcome."""
    sample = examples if sample_size >= len(examples) else rng.sample(examples, sample_size)

    connection = sqlite3.connect(database_path)
    counts: collections.Counter[str] = collections.Counter()
    errors: list[dict[str, str]] = []
    try:
        for example in sample:
            outcome = validate_example(connection, tables[example.table_id], example)
            counts[outcome.status] += 1
            if outcome.status == "error" and len(errors) < 5:
                errors.append({"sql": outcome.sql, "error": outcome.error or ""})
    finally:
        connection.close()

    total = sum(counts.values())
    return {
        "sampled": total,
        "usable": counts["usable"],
        "degenerate": counts["degenerate"],
        "error": counts["error"],
        "usable_rate": round(counts["usable"] / total, 4) if total else 0.0,
        "execution_error_rate": round(counts["error"] / total, 4) if total else 0.0,
        "error_examples": errors,
    }


# --------------------------------------------------------------------------
# Subsampling
# --------------------------------------------------------------------------


def stratum(example: WikiSQLExample) -> tuple[int, int]:
    """Stratify by query *shape*: aggregation type and condition count.

    These are what determine query difficulty in WikiSQL. Sampling uniformly
    at random would still approximate the distribution, but stratifying makes
    the match exact and, more importantly, guarantees the rare shapes (SUM/AVG
    with 3-4 conditions) are actually present in the training subsample rather
    than left to chance.
    """
    return (example.agg_index, min(len(example.conditions), 4))


def stratified_subsample(
    examples: list[WikiSQLExample], size: int, rng: random.Random
) -> list[WikiSQLExample]:
    if size >= len(examples):
        return list(examples)

    groups: dict[tuple[int, int], list[WikiSQLExample]] = collections.defaultdict(list)
    for example in examples:
        groups[stratum(example)].append(example)

    selected: list[WikiSQLExample] = []
    # Largest-remainder allocation, so proportions are preserved and the total
    # lands exactly on `size` rather than drifting with rounding.
    quotas = {key: len(group) * size / len(examples) for key, group in groups.items()}
    floors = {key: int(quota) for key, quota in quotas.items()}
    for key, count in floors.items():
        selected.extend(rng.sample(groups[key], min(count, len(groups[key]))))

    shortfall = size - len(selected)
    if shortfall > 0:
        remainders = sorted(
            groups, key=lambda k: quotas[k] - floors[k], reverse=True
        )
        chosen = {id(e) for e in selected}
        for key in remainders:
            if shortfall == 0:
                break
            for candidate in rng.sample(groups[key], len(groups[key])):
                if id(candidate) not in chosen:
                    selected.append(candidate)
                    chosen.add(id(candidate))
                    shortfall -= 1
                    break

    rng.shuffle(selected)
    return selected


def shape_distribution(examples: list[WikiSQLExample]) -> dict[str, float]:
    counts = collections.Counter(stratum(e) for e in examples)
    total = len(examples)
    return {f"agg{a}_cond{c}": round(n / total, 4) for (a, c), n in sorted(counts.items())}


# --------------------------------------------------------------------------
# Record writing
# --------------------------------------------------------------------------


def write_records(
    path: Path,
    examples: list[WikiSQLExample],
    tables: dict[str, WikiSQLTable],
) -> int:
    """Write prompt-ready records.

    Prompt *formatting* is deliberately not done here -- it belongs to the
    eval harness in Phase 2, which must apply an identical template to the
    baseline and fine-tuned runs. This file carries the raw ingredients only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            table = tables[example.table_id]
            record = {
                "table_id": table.id,
                "table_name": table.name,
                "columns": table.header,
                "column_types": table.types,
                "question": example.question,
                "gold_sql": render_sql(table, example),
                "agg_index": example.agg_index,
                "condition_count": len(example.conditions),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def select_usable(
    database_path: Path,
    tables: dict[str, WikiSQLTable],
    examples: list[WikiSQLExample],
    wanted: int,
    rng: random.Random,
) -> tuple[list[WikiSQLExample], int]:
    """Draw `wanted` examples whose ground truth returns a non-degenerate result."""
    shuffled = list(examples)
    rng.shuffle(shuffled)

    connection = sqlite3.connect(database_path)
    kept: list[WikiSQLExample] = []
    inspected = 0
    try:
        for example in shuffled:
            inspected += 1
            outcome = validate_example(connection, tables[example.table_id], example)
            if outcome.status == "usable":
                kept.append(example)
                if len(kept) == wanted:
                    break
    finally:
        connection.close()
    return kept, inspected


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-subsample", type=int, default=6000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--validate-sample", type=int, default=2000)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 1 gate -- WikiSQL materialisation into executable SQLite")
    print("=" * 72)

    rng = random.Random(SEED)
    raw_dir = ensure_raw_data(force=args.force_download)

    section("2. Load the official splits")
    splits = {name: load_split(raw_dir, name) for name in ("train", "dev", "test")}

    section("3. Materialise tables into SQLite")
    materialisation = {}
    for name, (tables, examples) in splits.items():
        stats = materialize_split(tables.values(), TABLES_DIR / f"{name}.db")
        materialisation[name] = stats.as_dict()
        print(
            f"  {name:5}: {stats.tables_written:6,} tables  "
            f"{stats.rows_written:8,} rows  {stats.database_bytes / 1024**2:7.1f} MiB"
        )

        # Excluding tables is only safe if nothing references them. Asserted
        # here rather than assumed, so a future corpus change cannot quietly
        # drop examples on the floor.
        if stats.skipped_table_ids:
            skipped = set(stats.skipped_table_ids)
            orphaned = [e for e in examples if e.table_id in skipped]
            print(
                f"         skipped {len(skipped)} table(s) with case-colliding "
                f"headers; {len(orphaned)} example(s) reference them"
            )
            if orphaned:
                raise SystemExit(
                    f"{name}: {len(orphaned)} examples reference skipped tables. "
                    "Exclusion is no longer lossless -- revisit the strategy in "
                    "materialize.has_colliding_headers()."
                )
            materialisation[name]["orphaned_examples"] = 0

    section("4. Validate ground-truth SQL by executing it")
    validation = {}
    for name, (tables, examples) in splits.items():
        result = validate_split(
            TABLES_DIR / f"{name}.db", tables, examples, args.validate_sample, rng
        )
        validation[name] = result
        print(
            f"  {name:5}: {result['usable']:5,}/{result['sampled']:,} usable "
            f"({result['usable_rate']:.1%})  degenerate={result['degenerate']:4,}  "
            f"errors={result['error']}"
        )
        for err in result["error_examples"]:
            print(f"         ! {err['error']}  |  {err['sql'][:70]}")

    section("5. Build the training subsample and evaluation set")
    train_tables, train_examples = splits["train"]
    test_tables, test_examples = splits["test"]

    subsample = stratified_subsample(train_examples, args.train_subsample, rng)
    train_path = PROCESSED_DIR / "train_subsample.jsonl"
    written_train = write_records(train_path, subsample, train_tables)
    print(f"  train subsample : {written_train:,} records -> {train_path.name}")

    eval_examples, inspected = select_usable(
        TABLES_DIR / "test.db", test_tables, test_examples, args.eval_size, rng
    )
    eval_path = PROCESSED_DIR / "test_eval.jsonl"
    written_eval = write_records(eval_path, eval_examples, test_tables)
    print(
        f"  evaluation set  : {written_eval:,} records -> {eval_path.name} "
        f"(drawn from {inspected:,} inspected; degenerate ground truth excluded)"
    )

    full_dist = shape_distribution(train_examples)
    sub_dist = shape_distribution(subsample)
    max_drift = max(
        abs(sub_dist.get(k, 0.0) - v) for k, v in full_dist.items()
    )
    print(f"  max stratum drift vs full train split: {max_drift:.4f}")

    # ---------------------------------------------------------------- report
    passed = (
        all(v["error"] == 0 for v in validation.values())
        and all(v["usable"] >= 20 for v in validation.values())
        and written_eval == args.eval_size
    )

    report = {
        "phase": 1,
        "purpose": "Materialise WikiSQL into executable SQLite and validate ground truth",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_result": "PASS" if passed else "FAIL",
        "seed": SEED,
        "source_url": DATA_URL,
        "split_sizes": {k: len(v[1]) for k, v in splits.items()},
        "materialisation": materialisation,
        "ground_truth_validation": validation,
        "train_subsample": {
            "size": written_train,
            "path": str(train_path.relative_to(REPO_ROOT)),
            "max_stratum_drift_vs_full_split": round(max_drift, 5),
        },
        "evaluation_set": {
            "size": written_eval,
            "path": str(eval_path.relative_to(REPO_ROOT)),
            "candidates_inspected": inspected,
            "selection_rule": "ground-truth SQL must return a non-degenerate result",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"PHASE 1: {'PASS' if passed else 'FAIL'}")
    print(f"Report written to {REPORT_PATH}")
    print("=" * 72)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
