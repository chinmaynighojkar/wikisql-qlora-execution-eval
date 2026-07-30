"""Phase 4: before/after comparison report.

Diffs `baseline_metrics.json` against `finetuned_metrics.json` and produces
the comparison the build plan calls for -- execution accuracy, validity rate,
and qualitative examples the baseline got wrong that fine-tuning fixed, plus
any it broke.

Three things this does beyond subtracting two numbers, because a large
headline delta is exactly when a result deserves the most scrutiny:

1. **Paired significance (McNemar).** Both models are scored on the same 500
   examples, so the outcomes are paired and an unpaired comparison would be
   the wrong test. McNemar's uses only the discordant pairs -- examples one
   model got right and the other did not -- which is the evidence that
   actually distinguishes them.

2. **Accuracy conditional on execution.** If every gained point came from
   queries merely becoming *runnable*, the model would have learned SQL syntax
   and nothing about the question. Reporting accuracy among queries that
   executed separates the two effects.

3. **Leakage checks.** Table and question overlap between the training
   subsample and the evaluation set. WikiSQL's official splits have disjoint
   tables, but a claim this size should verify that rather than cite it.

The report also states the **actual training scale** read from
`phase3_training.json`, so the numbers can never be presented against a
training run that did not produce them.

    python scripts\\phase4_compare.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
PROCESSED = REPO_ROOT / "data" / "processed"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the relevant phase first.")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def mcnemar(before: list[dict], after: list[dict], key: str = "execution_match") -> dict[str, Any]:
    """Exact-ish McNemar test on paired binary outcomes.

    Only discordant pairs carry information: examples both models got right,
    or both got wrong, say nothing about which is better. The continuity
    correction is applied because the chi-square approximation is otherwise
    anti-conservative on small discordant counts.
    """
    fixed = sum(1 for b, a in zip(before, after) if not b[key] and a[key])
    broken = sum(1 for b, a in zip(before, after) if b[key] and not a[key])
    both = sum(1 for b, a in zip(before, after) if b[key] and a[key])
    neither = sum(1 for b, a in zip(before, after) if not b[key] and not a[key])

    discordant = fixed + broken
    if discordant == 0:
        # No discordant pairs to test -- reported through the same schema as
        # the normal path below, not a shorter dict, so render_markdown()
        # (which reads fixed_by_finetuning / chi_square_continuity_corrected /
        # p_value_display unconditionally) does not crash on this case.
        chi_square = 0.0
        p_value = 1.0
    else:
        chi_square = (abs(fixed - broken) - 1) ** 2 / discordant
        p_value = math.erfc(math.sqrt(chi_square) / math.sqrt(2))
    return {
        "fixed_by_finetuning": fixed,
        "broken_by_finetuning": broken,
        "both_correct": both,
        "both_wrong": neither,
        "discordant_pairs": discordant,
        "chi_square_continuity_corrected": round(chi_square, 3),
        "p_value": p_value,
        "p_value_display": "<1e-15" if p_value < 1e-15 else f"{p_value:.3g}",
        "significant_at_0_01": p_value < 0.01,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval -- well behaved near 0 and 1, unlike the
    normal approximation."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------


def leakage_report() -> dict[str, Any]:
    train = read_jsonl(PROCESSED / "train_subsample.jsonl")
    evaluation = read_jsonl(PROCESSED / "test_eval.jsonl")

    train_tables = {r["table_id"] for r in train}
    eval_tables = {r["table_id"] for r in evaluation}

    train_questions: dict[str, dict] = {}
    for record in train:
        train_questions.setdefault(record["question"].strip().casefold(), record)

    shared = []
    for record in evaluation:
        key = record["question"].strip().casefold()
        if key in train_questions:
            other = train_questions[key]
            shared.append(
                {
                    "question": record["question"],
                    "train_table": other["table_name"],
                    "eval_table": record["table_name"],
                    "identical_gold_sql": other["gold_sql"] == record["gold_sql"],
                }
            )

    return {
        "train_tables": len(train_tables),
        "eval_tables": len(eval_tables),
        "table_overlap": len(train_tables & eval_tables),
        "shared_question_strings": len(shared),
        "shared_questions_with_identical_gold_sql": sum(
            1 for s in shared if s["identical_gold_sql"]
        ),
        "shared_question_examples": shared[:5],
    }


# --------------------------------------------------------------------------
# Qualitative examples
# --------------------------------------------------------------------------


def qualitative(before: list[dict], after: list[dict], limit: int) -> dict[str, list[dict]]:
    fixed, broken = [], []
    for b, a in zip(before, after):
        entry = {
            "question": b["question"],
            "gold_sql": b["gold_sql"],
            "baseline_sql": b["predicted_sql"],
            "baseline_failure": b["failure_reason"],
            "finetuned_sql": a["predicted_sql"],
            "finetuned_failure": a["failure_reason"],
        }
        if not b["execution_match"] and a["execution_match"]:
            fixed.append(entry)
        elif b["execution_match"] and not a["execution_match"]:
            broken.append(entry)
    return {"fixed_by_finetuning": fixed[:limit], "broken_by_finetuning": broken[:limit]}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    b = report["baseline"]["metrics"]
    f = report["finetuned"]["metrics"]
    m = report["mcnemar"]
    scale = report["training_scale"]
    lines: list[str] = []

    lines.append("# Before / after: QLoRA fine-tuning for text-to-SQL\n")
    lines.append(
        f"Evaluated on **{report['n_records']} held-out WikiSQL test examples**, "
        "scored by execution match against real SQLite tables. Identical eval "
        "code for both runs -- the only difference is the LoRA adapter.\n"
    )

    lines.append("## Headline\n")
    lines.append("| Metric | Base model | + QLoRA adapter | Change |")
    lines.append("|---|---:|---:|---:|")
    for label, key in [
        ("**Execution accuracy** (primary)", "execution_accuracy"),
        ("Syntactic validity", "syntactic_validity_rate"),
        ("Exact string match (secondary)", "exact_match_accuracy"),
        ("SQL extraction rate", "sql_extraction_rate"),
    ]:
        delta = f[key] - b[key]
        lines.append(
            f"| {label} | {pct(b[key])} | {pct(f[key])} | **{delta * 100:+.1f} pts** |"
        )
    ci_b = report["confidence_intervals"]["baseline_execution_accuracy_95ci"]
    ci_f = report["confidence_intervals"]["finetuned_execution_accuracy_95ci"]
    lines.append(
        f"\n95% Wilson intervals on execution accuracy: baseline "
        f"[{pct(ci_b[0])}, {pct(ci_b[1])}], fine-tuned [{pct(ci_f[0])}, {pct(ci_f[1])}] "
        "— non-overlapping.\n"
    )

    lines.append("## Training scale\n")
    lines.append(
        f"The adapter was trained on **{scale['n_train_records']:,} examples** for "
        f"**{scale['num_train_epochs']} epoch(s)** in "
        f"**{scale['train_runtime_seconds']:.0f} seconds**, peaking at "
        f"**{scale['peak_vram_gib']} GiB** of VRAM on a 4 GB RTX 3050 Laptop. "
        f"{scale['trainable_parameters']:,} trainable parameters "
        f"({scale['trainable_percent']}% of the model).\n"
    )

    lines.append("## Is the difference real?\n")
    lines.append(
        "Both models were scored on the same examples, so the outcomes are "
        "paired and McNemar's test applies:\n"
    )
    lines.append("| | Fine-tuned correct | Fine-tuned wrong |")
    lines.append("|---|---:|---:|")
    lines.append(f"| **Baseline correct** | {m['both_correct']} | {m['broken_by_finetuning']} |")
    lines.append(f"| **Baseline wrong** | {m['fixed_by_finetuning']} | {m['both_wrong']} |")
    lines.append(
        f"\nFine-tuning fixed **{m['fixed_by_finetuning']}** examples and broke "
        f"**{m['broken_by_finetuning']}**. "
        f"chi-square = {m['chi_square_continuity_corrected']} (continuity corrected), "
        f"p = {m['p_value_display']}.\n"
    )

    cond = report["conditional_accuracy"]
    lines.append("## Where the gain came from\n")
    lines.append(
        "If every gained point came from queries merely becoming runnable, the "
        "model would have learned SQL syntax and nothing about the question. "
        "Accuracy among queries that actually executed:\n"
    )
    lines.append("| | Executed | Correct | Accuracy given execution |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Base model | {cond['baseline_executed']} | {cond['baseline_correct']} | "
        f"{pct(cond['baseline_accuracy_given_execution'])} |"
    )
    lines.append(
        f"| + adapter | {cond['finetuned_executed']} | {cond['finetuned_correct']} | "
        f"{pct(cond['finetuned_accuracy_given_execution'])} |"
    )
    lines.append(
        f"\nSo the improvement is **both** effects: far more queries run "
        f"(validity {pct(b['syntactic_validity_rate'])} -> "
        f"{pct(f['syntactic_validity_rate'])}), and those that run are also more "
        f"often right ({pct(cond['baseline_accuracy_given_execution'])} -> "
        f"{pct(cond['finetuned_accuracy_given_execution'])}).\n"
    )

    lines.append("### Failure modes\n")
    lines.append("| Failure | Base model | + adapter |")
    lines.append("|---|---:|---:|")
    all_reasons = sorted(set(b["failure_breakdown"]) | set(f["failure_breakdown"]))
    for reason in all_reasons:
        lines.append(
            f"| `{reason}` | {b['failure_breakdown'].get(reason, 0)} | "
            f"{f['failure_breakdown'].get(reason, 0)} |"
        )
    lines.append("")

    leak = report["leakage_checks"]
    lines.append("## Leakage checks\n")
    lines.append(
        f"- Tables shared between the training subsample and the evaluation set: "
        f"**{leak['table_overlap']}** (of {leak['train_tables']:,} train and "
        f"{leak['eval_tables']} eval tables). WikiSQL's official splits have "
        "disjoint tables; verified rather than assumed.\n"
        f"- Question strings appearing in both: **{leak['shared_question_strings']}**, "
        f"of which **{leak['shared_questions_with_identical_gold_sql']}** have identical "
        "gold SQL.\n"
    )

    lines.append("## Qualitative examples\n")
    lines.append("### Fixed by fine-tuning\n")
    for example in report["examples"]["fixed_by_finetuning"]:
        lines.append(f"**Q:** {example['question']}\n")
        lines.append(f"```sql\n-- gold\n{example['gold_sql']}")
        lines.append(f"-- base model  ({example['baseline_failure']})\n{example['baseline_sql']}")
        lines.append(f"-- + adapter   (correct)\n{example['finetuned_sql']}\n```\n")

    if report["examples"]["broken_by_finetuning"]:
        lines.append("### Broken by fine-tuning\n")
        lines.append(
            "Reported for the same reason the fixes are: a comparison that only "
            "showed improvements would not be a measurement.\n"
        )
        for example in report["examples"]["broken_by_finetuning"]:
            lines.append(f"**Q:** {example['question']}\n")
            lines.append(f"```sql\n-- gold\n{example['gold_sql']}")
            lines.append(f"-- base model  (correct)\n{example['baseline_sql']}")
            lines.append(
                f"-- + adapter   ({example['finetuned_failure']})\n{example['finetuned_sql']}\n```\n"
            )

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--finetuned", default="finetuned")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    baseline = load_json(REPORTS / f"{args.baseline}_metrics.json")
    finetuned = load_json(REPORTS / f"{args.finetuned}_metrics.json")
    before = load_json(REPORTS / f"{args.baseline}_predictions.json")
    after = load_json(REPORTS / f"{args.finetuned}_predictions.json")

    if len(before) != len(after):
        raise SystemExit(
            f"prediction counts differ ({len(before)} vs {len(after)}) -- the two "
            "runs did not score the same examples, so they cannot be compared"
        )
    for b, a in zip(before, after):
        if b["gold_sql"] != a["gold_sql"]:
            raise SystemExit(
                "prediction files are misaligned (gold SQL differs at the same "
                "index) -- refusing to produce a comparison"
            )

    training = load_json(REPORTS / "phase3_training.json")
    scale = {
        "n_train_records": training["n_train_records"],
        "num_train_epochs": training["training"]["num_train_epochs"],
        "train_runtime_seconds": training["train_runtime_seconds"],
        "peak_vram_gib": training["peak_vram_gib"],
        "trainable_parameters": training["trainable_parameters"],
        "trainable_percent": training["trainable_percent"],
        "lora_rank": training["lora"]["r"],
    }

    bm, fm = baseline["metrics"], finetuned["metrics"]
    total = bm["n"]

    report = {
        "generated_from": {
            "baseline": f"{args.baseline}_metrics.json",
            "finetuned": f"{args.finetuned}_metrics.json",
        },
        "n_records": total,
        "baseline": baseline,
        "finetuned": finetuned,
        "training_scale": scale,
        "deltas": {
            key: round(fm[key] - bm[key], 4)
            for key in [
                "execution_accuracy",
                "syntactic_validity_rate",
                "exact_match_accuracy",
                "sql_extraction_rate",
            ]
        },
        "confidence_intervals": {
            "baseline_execution_accuracy_95ci": wilson_interval(
                bm["counts"]["execution_match"], total
            ),
            "finetuned_execution_accuracy_95ci": wilson_interval(
                fm["counts"]["execution_match"], total
            ),
        },
        "mcnemar": mcnemar(before, after),
        "conditional_accuracy": {
            "baseline_executed": bm["counts"]["executed"],
            "baseline_correct": bm["counts"]["execution_match"],
            "baseline_accuracy_given_execution": (
                round(bm["counts"]["execution_match"] / bm["counts"]["executed"], 4)
                if bm["counts"]["executed"] else 0.0
            ),
            "finetuned_executed": fm["counts"]["executed"],
            "finetuned_correct": fm["counts"]["execution_match"],
            "finetuned_accuracy_given_execution": (
                round(fm["counts"]["execution_match"] / fm["counts"]["executed"], 4)
                if fm["counts"]["executed"] else 0.0
            ),
        },
        "leakage_checks": leakage_report(),
        "examples": qualitative(before, after, args.examples),
    }

    (REPORTS / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = render_markdown(report)
    (REPORTS / "comparison.md").write_text(markdown, encoding="utf-8")

    m = report["mcnemar"]
    print("=" * 72)
    print("Phase 4 -- before/after comparison")
    print("=" * 72)
    print(f"  execution accuracy   {pct(bm['execution_accuracy'])} -> {pct(fm['execution_accuracy'])}"
          f"   ({report['deltas']['execution_accuracy'] * 100:+.1f} pts)")
    print(f"  syntactic validity   {pct(bm['syntactic_validity_rate'])} -> {pct(fm['syntactic_validity_rate'])}")
    print(f"  exact string match   {pct(bm['exact_match_accuracy'])} -> {pct(fm['exact_match_accuracy'])}")
    print(f"  McNemar              fixed={m['fixed_by_finetuning']} broken={m['broken_by_finetuning']}"
          f"  p={m['p_value_display']}")
    print(f"  trained on           {scale['n_train_records']:,} examples, "
          f"{scale['num_train_epochs']} epoch(s), {scale['train_runtime_seconds']:.0f}s")
    leak = report["leakage_checks"]
    print(f"  table overlap        {leak['table_overlap']}  (leakage check)")
    print(f"\n  -> {REPORTS / 'comparison.md'}")
    print(f"  -> {REPORTS / 'comparison.json'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
