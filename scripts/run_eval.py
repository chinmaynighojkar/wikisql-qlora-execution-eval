"""Execution-match evaluation. Used by BOTH Phase 2 and Phase 4.

There is deliberately no `phase2_eval.py` and no `phase4_eval.py`. The build
plan's central claim depends on the baseline and the fine-tuned model being
scored by identical code, and the most reliable way to guarantee that is for
only one scoring program to exist. The sole difference between the two runs is
the `--adapter` flag:

    Phase 2 (baseline):
        python scripts/run_eval.py --name baseline

    Phase 4 (fine-tuned):
        python scripts/run_eval.py --name finetuned --adapter models/lora-adapter

Additional modes that need no GPU:

    --self-test           calibrate the harness against known-answer inputs
    --predictions-from F  re-score saved generations without re-running a model
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lora_text_to_sql.evaluate import (  # noqa: E402
    aggregate_metrics,
    breakdown_by,
    score_example,
)
from lora_text_to_sql.prompt import build_prompt  # noqa: E402

EVAL_RECORDS = REPO_ROOT / "data" / "processed" / "test_eval.jsonl"
TEST_DB = REPO_ROOT / "data" / "tables" / "test.db"
REPORTS_DIR = REPO_ROOT / "reports"


def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run scripts/phase1_prepare_data.py first."
        )
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return records[:limit] if limit else records


# --------------------------------------------------------------------------
# Self-test: calibrate the harness without a model
# --------------------------------------------------------------------------


def self_test(records: list[dict[str, Any]], connection: sqlite3.Connection) -> int:
    """Check the harness scores known-answer inputs correctly.

    A scoring harness that is never itself tested is the single easiest way to
    produce a confident, wrong before/after number. Each strategy below has a
    known correct outcome, so the harness's own accuracy can be measured
    rather than assumed.
    """
    import re

    def oracle(record):
        """Gold SQL, verbatim. Must score 100%."""
        return record["gold_sql"]

    def oracle_restyled(record):
        """Gold SQL rewritten to be semantically identical but textually different.

        This is the case that justifies the entire project premise, so it has
        to be a fair test. Lowercasing keywords is *not* sufficient: the
        exact-match metric casefolds, so it would absorb the change and both
        metrics would score 1.0, demonstrating nothing.

        Instead the identifier quoting style is switched from `"col"` to
        `[col]` -- both are valid SQLite, both are things a model plausibly
        emits, and casefolding cannot reconcile them. The expected outcome is
        the crux of the argument: execution match ~100%, exact match ~0%.

        Identifiers containing a square bracket are left double-quoted: 117 of
        the corpus's 26,531 tables have headers like `Kickoff [a ]` or
        `Richmond [Staten Is.]`, and `[Kickoff [a ]]` is a syntax error.
        (The harness caught exactly this when the strategy was first written
        and did not silently pass the malformed query -- which is the
        behaviour being verified.)
        """
        sql = record["gold_sql"]
        for column in record["columns"] + [record["table_name"]]:
            if "[" in column or "]" in column:
                continue
            sql = sql.replace(f'"{column}"', f"[{column}]")
        for keyword in ("SELECT", "FROM", "WHERE", "AND", "COUNT", "MAX", "MIN", "SUM", "AVG"):
            sql = sql.replace(keyword, keyword.lower())
        return sql

    def oracle_wrapped(record):
        """Gold SQL buried in markdown and prose -- tests extraction only."""
        return (
            "Sure! Here is the query you asked for:\n\n"
            f"```sql\n{record['gold_sql']};\n```\n\n"
            "This selects the relevant column."
        )

    def wrong_column(record):
        """A different column from the same table. Must score ~0%."""
        columns = record["columns"]
        gold = record["gold_sql"]
        alternative = next(
            (c for c in columns if f'"{c}"' not in gold), columns[-1]
        )
        return f'SELECT "{alternative}" FROM "{record["table_name"]}"'

    def unconditioned(record):
        """Drops the WHERE clause. Must score low: the adversarial case for a
        harness that only checks 'did it run?'."""
        return re.sub(r"\s+WHERE\s+.*$", "", record["gold_sql"], flags=re.IGNORECASE)

    def hallucinated_column(record):
        """A plausible column name that does not exist, double-quoted.

        Without the identifier guard in `evaluate.find_unknown_identifiers`,
        SQLite reinterprets this as a string literal and the query executes
        cleanly -- so `syntactic_validity_rate` would read ~100% for queries
        that are entirely wrong. Must score 0 on both metrics.
        """
        return f'SELECT "TotalScoreValue" FROM "{record["table_name"]}"'

    def garbage(record):
        return "I am not able to answer that question."

    strategies = [
        ("gold verbatim", oracle, "execution", 0.99, 1.0),
        ("gold restyled", oracle_restyled, "execution", 0.99, 1.0),
        ("gold in markdown + prose", oracle_wrapped, "execution", 0.99, 1.0),
        ("gold restyled", oracle_restyled, "exact", 0.0, 0.05),
        ("wrong column", wrong_column, "execution", 0.0, 0.10),
        ("WHERE clause removed", unconditioned, "execution", 0.0, 0.15),
        ("hallucinated column", hallucinated_column, "execution", 0.0, 0.0),
        ("hallucinated column", hallucinated_column, "validity", 0.0, 0.0),
        ("non-SQL text", garbage, "execution", 0.0, 0.0),
        ("non-SQL text", garbage, "validity", 0.0, 0.0),
    ]

    print(f"\nHarness self-test on {len(records)} real evaluation records")
    print("-" * 72)
    print(f"{'strategy':28} {'metric':10} {'observed':>9}  {'expected range':>16}  result")

    failures = 0
    for label, strategy, metric, low, high in strategies:
        results = [score_example(connection, r, strategy(r)) for r in records]
        metrics = aggregate_metrics(results)
        observed = {
            "execution": metrics["execution_accuracy"],
            "exact": metrics["exact_match_accuracy"],
            "validity": metrics["syntactic_validity_rate"],
        }[metric]
        ok = low <= observed <= high
        failures += not ok
        print(
            f"{label:28} {metric:10} {observed:9.4f}  "
            f"{f'[{low:.2f}, {high:.2f}]':>16}  {'PASS' if ok else 'FAIL'}"
        )

    print("-" * 72)
    if failures:
        print(f"SELF-TEST FAILED: {failures} expectation(s) violated")
    else:
        print("SELF-TEST PASSED: the harness scores known-answer inputs correctly")
    return 1 if failures else 0


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="baseline", help="output name, e.g. baseline / finetuned")
    parser.add_argument("--adapter", default=None, help="path to a trained LoRA adapter (Phase 4)")
    parser.add_argument("--model", default=None, help="override the model id")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N records")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--self-test", action="store_true", help="calibrate the harness, no GPU needed")
    parser.add_argument("--predictions-from", default=None, help="re-score saved generations")
    args = parser.parse_args()

    records = load_records(EVAL_RECORDS, args.limit)
    if not TEST_DB.exists():
        raise SystemExit(f"{TEST_DB} not found. Run scripts/phase1_prepare_data.py first.")
    connection = sqlite3.connect(f"file:{TEST_DB}?mode=ro", uri=True)

    if args.self_test:
        return self_test(records, connection)

    print("=" * 72)
    print(f"Execution-match evaluation -- {args.name}")
    print(f"  records : {len(records)}")
    print(f"  adapter : {args.adapter or 'none (base model)'}")
    print("=" * 72)

    # ------------------------------------------------------------- generate
    if args.predictions_from:
        saved = json.loads(Path(args.predictions_from).read_text(encoding="utf-8"))
        outputs = [row["raw_output"] for row in saved]
        if len(outputs) != len(records):
            raise SystemExit(
                f"{len(outputs)} saved generations vs {len(records)} records -- mismatch"
            )
        elapsed = 0.0
    else:
        from lora_text_to_sql.generation import (
            generate_batched,
            load_config,
            load_model_and_tokenizer,
        )

        config = load_config()
        print("\nLoading model...")
        model, tokenizer = load_model_and_tokenizer(config, args.adapter, args.model)

        prompts = [build_prompt(tokenizer, record) for record in records]
        print(f"Generating ({len(prompts)} prompts, batch size {args.batch_size})...")
        started = time.perf_counter()
        outputs = list(
            generate_batched(
                model,
                tokenizer,
                prompts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=config["runtime"]["max_seq_length"],
            )
        )
        elapsed = round(time.perf_counter() - started, 1)
        print(f"Generation finished in {elapsed}s")

    # ---------------------------------------------------------------- score
    results = [
        score_example(connection, record, output)
        for record, output in zip(records, outputs)
    ]
    metrics = aggregate_metrics(results)
    metrics["by_condition_count"] = breakdown_by(results, records, "condition_count")
    metrics["by_agg_index"] = breakdown_by(results, records, "agg_index")

    report = {
        "name": args.name,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "adapter": args.adapter,
        "model": args.model,
        "eval_records": str(EVAL_RECORDS.relative_to(REPO_ROOT)),
        "n_records": len(records),
        "decoding": {"strategy": "greedy", "max_new_tokens": args.max_new_tokens},
        "generation_seconds": elapsed,
        "metrics": metrics,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / f"{args.name}_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions_path = REPORTS_DIR / f"{args.name}_predictions.json"
    predictions_path.write_text(
        json.dumps([r.as_dict() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # --------------------------------------------------------------- report
    print("\n" + "=" * 72)
    print(f"  execution accuracy      {metrics['execution_accuracy']:.2%}   <- primary")
    print(f"  syntactic validity      {metrics['syntactic_validity_rate']:.2%}")
    print(f"  exact string match      {metrics['exact_match_accuracy']:.2%}   (secondary)")
    print(f"  SQL extraction rate     {metrics['sql_extraction_rate']:.2%}")
    if metrics["failure_breakdown"]:
        print("  failures:")
        for reason, count in metrics["failure_breakdown"].items():
            print(f"    {reason:16} {count}")
    print(f"\n  metrics     -> {metrics_path}")
    print(f"  predictions -> {predictions_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
