"""Data-efficiency study: how much fine-tuning data does QLoRA actually need?

Trains the same LoRA configuration at several training-set sizes and scores
each with the identical Phase 2/4 harness, producing an execution-accuracy
curve rather than a single before/after point.

The question is worth asking because the smoke run answered it accidentally:
200 examples and 102 seconds of training moved execution accuracy from 39.8%
to 76.0%. If most of the achievable gain arrives that early, that is a more
useful finding than a single number at 6,000 examples -- and it is the kind of
thing a practitioner actually needs to know before budgeting a fine-tune.

Design notes:

- **Epochs are held constant, data varies.** This is the standard framing of a
  data-efficiency curve, and it does mean the number of optimiser steps scales
  with the data. Data quantity and training duration are therefore
  deliberately confounded -- reported honestly rather than papered over,
  because "more data, same recipe" is what the decision actually looks like
  in practice.
- **Each run is a separate subprocess.** Loading and discarding three
  quantised models in one process fragments a 4 GB card badly; a fresh
  process guarantees a clean allocator each time.
- **Resumable.** Completed stages are skipped, so an interrupted study can be
  restarted without repeating hours of training.

    python scripts\\run_scaling_study.py
    python scripts\\run_scaling_study.py --sizes 200 1000 6000
    python scripts\\run_scaling_study.py --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from lora_text_to_sql.provenance import capture as capture_provenance

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
MODELS = REPO_ROOT / "models"
PYTHON = sys.executable

DEFAULT_SIZES = [200, 1000, 6000]


def adapter_dir(size: int) -> Path:
    return MODELS / f"lora-{size}"


def training_report(size: int) -> Path:
    return REPORTS / f"phase3_training_{size}.json"


def metrics_path(size: int) -> Path:
    return REPORTS / f"finetuned{size}_metrics.json"


def run(command: list[str], label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print("  $ " + " ".join(command[1:]))
    started = time.perf_counter()
    result = subprocess.run(command, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        # Phase 3 exits non-zero on FAIL/INDETERMINATE, which is a verdict about
        # the run rather than a crash, so it is reported and tolerated here.
        print(f"  (exit code {result.returncode} after {elapsed / 60:.1f} min)")
    else:
        print(f"  completed in {elapsed / 60:.1f} min")


def train_stage(size: int, epochs: float, force: bool) -> None:
    if adapter_dir(size).exists() and training_report(size).exists() and not force:
        print(f"\n[skip] training at n={size} (adapter and report already exist)")
        return
    run(
        [
            PYTHON, "scripts/phase3_train.py",
            "--limit", str(size),
            "--epochs", str(epochs),
            "--output", str(adapter_dir(size)),
            "--report", str(training_report(size)),
        ],
        f"TRAIN  n={size}  epochs={epochs}",
    )


def eval_stage(size: int, batch_size: int, force: bool) -> None:
    metrics, report = metrics_path(size), training_report(size)
    # Skipping on metrics existence alone is not enough: if training was just
    # re-run (e.g. at a different epoch count) but a stale metrics file from
    # an earlier run wasn't deleted, the skip would silently pair fresh
    # training with an old evaluation. Comparing mtimes catches that case.
    stale = metrics.exists() and report.exists() and metrics.stat().st_mtime < report.stat().st_mtime
    if metrics.exists() and not force and not stale:
        print(f"\n[skip] evaluation at n={size} (metrics already exist)")
        return
    if stale:
        print(f"\n[rerun] evaluation at n={size} (existing metrics predate the current training report)")
    run(
        [
            PYTHON, "scripts/run_eval.py",
            "--name", f"finetuned{size}",
            "--adapter", str(adapter_dir(size)),
            "--batch-size", str(batch_size),
        ],
        f"EVAL   n={size}",
    )


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate(sizes: list[int]) -> dict[str, Any]:
    baseline = json.loads((REPORTS / "baseline_metrics.json").read_text(encoding="utf-8"))
    points: list[dict[str, Any]] = [
        {
            "n_train_examples": 0,
            "label": "base model (no fine-tuning)",
            "execution_accuracy": baseline["metrics"]["execution_accuracy"],
            "syntactic_validity_rate": baseline["metrics"]["syntactic_validity_rate"],
            "exact_match_accuracy": baseline["metrics"]["exact_match_accuracy"],
            "train_runtime_seconds": 0.0,
            "peak_vram_gib": None,
        }
    ]

    for size in sizes:
        if not metrics_path(size).exists():
            print(f"  (no metrics for n={size}; skipping)")
            continue
        metrics = json.loads(metrics_path(size).read_text(encoding="utf-8"))["metrics"]
        training: dict[str, Any] = {}
        if training_report(size).exists():
            training = json.loads(training_report(size).read_text(encoding="utf-8"))
        points.append(
            {
                "n_train_examples": size,
                "label": f"QLoRA, {size:,} examples",
                "execution_accuracy": metrics["execution_accuracy"],
                "syntactic_validity_rate": metrics["syntactic_validity_rate"],
                "exact_match_accuracy": metrics["exact_match_accuracy"],
                "train_runtime_seconds": training.get("train_runtime_seconds"),
                "peak_vram_gib": training.get("peak_vram_gib"),
                "optimizer_steps": training.get("optimizer_steps"),
                "final_train_loss": training.get("final_train_loss"),
            }
        )

    base = points[0]["execution_accuracy"]
    best = max(p["execution_accuracy"] for p in points)
    for point in points:
        gain = point["execution_accuracy"] - base
        total_gain = best - base
        point["gain_over_baseline_pts"] = round(gain * 100, 1)
        point["share_of_best_gain"] = round(gain / total_gain, 3) if total_gain else 0.0

    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": capture_provenance(REPO_ROOT),
        "eval_records": baseline["n_records"],
        "note": (
            "Epochs held constant while training-set size varies, so optimiser "
            "steps scale with data. Data quantity and training duration are "
            "confounded by design; this mirrors the practical decision."
        ),
        "points": points,
    }


def render(curve: dict[str, Any]) -> str:
    lines = ["# Data-efficiency curve\n"]
    lines.append(
        f"How execution accuracy varies with the amount of fine-tuning data, "
        f"scored on the same {curve['eval_records']} held-out WikiSQL test "
        "examples with identical eval code at every point.\n"
    )
    lines.append("| Train examples | Execution accuracy | Validity | Exact match | Gain | Share of total gain | Train time |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for p in curve["points"]:
        runtime = p.get("train_runtime_seconds")
        runtime_str = "—" if not runtime else (
            f"{runtime:.0f}s" if runtime < 120 else f"{runtime / 60:.0f} min"
        )
        lines.append(
            f"| {p['n_train_examples']:,} | **{p['execution_accuracy'] * 100:.1f}%** | "
            f"{p['syntactic_validity_rate'] * 100:.1f}% | "
            f"{p['exact_match_accuracy'] * 100:.1f}% | "
            f"{p['gain_over_baseline_pts']:+.1f} pts | "
            f"{p['share_of_best_gain'] * 100:.0f}% | {runtime_str} |"
        )
    lines.append(f"\n> {curve['note']}\n")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="re-run completed stages")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    sizes = sorted(args.sizes)

    if not args.aggregate_only:
        print("Data-efficiency study")
        print(f"  sizes  : {sizes}")
        print(f"  epochs : {args.epochs} (held constant)")
        for size in sizes:
            train_stage(size, args.epochs, args.force)
            eval_stage(size, args.batch_size, args.force)

    print(f"\n{'=' * 72}\nAGGREGATE\n{'=' * 72}")
    curve = aggregate(sizes)
    (REPORTS / "scaling_curve.json").write_text(
        json.dumps(curve, indent=2), encoding="utf-8"
    )
    markdown = render(curve)
    (REPORTS / "scaling_curve.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"  -> {REPORTS / 'scaling_curve.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
