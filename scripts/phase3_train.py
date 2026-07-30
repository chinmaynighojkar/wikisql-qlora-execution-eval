"""Phase 3 gate: QLoRA fine-tuning.

Pass criteria from the build plan: training loss decreases, and the resulting
adapter produces valid SQL on 3 held-out examples.

    python scripts\\phase3_train.py
    python scripts\\phase3_train.py --limit 200 --epochs 1   # fast smoke run

Held-out examples come from the **dev** split. They are never drawn from the
train subsample (that would measure memorisation) and never from the Phase 2/4
test set (that would leak the evaluation set into a go/no-go decision made
during training).

Outputs:
    models/lora-adapter/            the trained adapter, loaded by Phase 4
    reports/phase3_training.json    loss curve, config, smoke-test results
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from lora_text_to_sql.dataset import (
    build_sft_dataset,
    load_records,
    token_length_stats,
)
from lora_text_to_sql.evaluate import score_example
from lora_text_to_sql.prompt import build_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIN_RECORDS = REPO_ROOT / "data" / "processed" / "train_subsample.jsonl"
DEV_DB = REPO_ROOT / "data" / "tables" / "dev.db"
RAW_DEV = REPO_ROOT / "data" / "raw" / "data"
ADAPTER_DIR = REPO_ROOT / "models" / "lora-adapter"
REPORT_PATH = REPO_ROOT / "reports" / "phase3_training.json"
TRAINING_CONFIG = REPO_ROOT / "configs" / "training.yaml"


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def assess_loss_trend(losses: list[float], min_points: int = 4) -> tuple[bool | None, str]:
    """Decide whether training loss fell, robustly.

    Comparing `losses[0]` to `losses[-1]` is wrong twice over. With a short
    run it can compare a value to itself -- a 200-record smoke run at
    `gradient_accumulation_steps=16` produces 12 optimiser steps, which at
    `logging_steps=10` logs a single point, making the comparison vacuous and
    reporting FAIL for a run that trained fine. And on a longer run, two
    single noisy points is a poor estimate of a trend.

    Instead: require enough logged points to say anything at all, then compare
    the mean of the first quarter against the mean of the last quarter.

    Returns `(None, reason)` when there is not enough data -- which is
    *indeterminate*, not a failure, and is reported as such.
    """
    if len(losses) < min_points:
        return None, (
            f"only {len(losses)} logged loss point(s); at least {min_points} are "
            "needed to assess a trend. Lower logging_steps or train for longer."
        )
    window = max(1, len(losses) // 4)
    head = sum(losses[:window]) / window
    tail = sum(losses[-window:]) / window
    change = (tail - head) / head * 100 if head else 0.0
    return tail < head, (
        f"mean of first {window} logged step(s) = {head:.4f}, "
        f"last {window} = {tail:.4f} ({change:+.1f}%)"
    )


def build_holdout(size: int, seed: int) -> list[dict[str, Any]]:
    """Draw held-out examples from the dev split, with usable ground truth."""
    import random

    from lora_text_to_sql.materialize import validate_example
    from lora_text_to_sql.wikisql import load_examples, load_tables

    tables = load_tables(RAW_DEV / "dev.tables.jsonl")
    examples = list(load_examples(RAW_DEV / "dev.jsonl"))
    rng = random.Random(seed)
    rng.shuffle(examples)

    connection = sqlite3.connect(f"file:{DEV_DB}?mode=ro", uri=True)
    holdout: list[dict[str, Any]] = []
    try:
        for example in examples:
            table = tables[example.table_id]
            outcome = validate_example(connection, table, example)
            if outcome.status != "usable":
                continue
            holdout.append(
                {
                    "table_id": table.id,
                    "table_name": table.name,
                    "columns": table.header,
                    "column_types": table.types,
                    "question": example.question,
                    "gold_sql": outcome.sql,
                    "agg_index": example.agg_index,
                    "condition_count": len(example.conditions),
                }
            )
            if len(holdout) == size:
                break
    finally:
        connection.close()
    return holdout


def run_smoke_test(model, tokenizer, holdout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate on held-out examples and score them with the Phase 2 harness."""
    import torch

    connection = sqlite3.connect(f"file:{DEV_DB}?mode=ro", uri=True)
    results = []
    try:
        for record in holdout:
            prompt = build_prompt(tokenizer, record)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            scored = score_example(connection, record, raw)
            results.append(scored.as_dict())
            print(f"\n  Q    : {record['question'][:70]}")
            print(f"  gold : {record['gold_sql'][:90]}")
            print(f"  pred : {(scored.predicted_sql or raw.strip())[:90]}")
            print(
                f"  -> executed={scored.executed}  match={scored.execution_match}"
                + (f"  ({scored.failure_reason})" if scored.failure_reason else "")
            )
    finally:
        connection.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N training records")
    parser.add_argument("--epochs", type=float, default=None, help="override num_train_epochs")
    parser.add_argument("--rank", type=int, default=None, help="override LoRA rank")
    parser.add_argument("--attention-only", action="store_true", help="drop the MLP projections (saves VRAM)")
    parser.add_argument("--logging-steps", type=int, default=None, help="override logging_steps")
    parser.add_argument("--output", default=str(ADAPTER_DIR))
    parser.add_argument(
        "--report",
        default=str(REPORT_PATH),
        help="where to write the training report (per-run, so a scaling study does not overwrite)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 3 gate -- QLoRA fine-tuning")
    print("=" * 72)

    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    from lora_text_to_sql.generation import load_config, load_model_and_tokenizer
    from lora_text_to_sql.seeding import seed_everything

    model_config = load_config()
    train_config = load_yaml(TRAINING_CONFIG)
    lora_settings = dict(train_config["lora"])
    settings = dict(train_config["training"])
    # SFTConfig(seed=...) below covers the trainer, but attaching the LoRA
    # adapter and any future non-deterministic step (a different sampler, a
    # random holdout draw) is not automatically inside that scope. Seeding
    # here once makes reproducibility a decision, not a side effect of the
    # current configuration.
    seed_everything(settings["seed"])

    if args.rank is not None:
        lora_settings["r"] = args.rank
        lora_settings["lora_alpha"] = args.rank * 2
    if args.attention_only:
        lora_settings["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.epochs is not None:
        settings["num_train_epochs"] = args.epochs

    # ------------------------------------------------------------------ data
    section("1. Load the training subsample")
    if not TRAIN_RECORDS.exists():
        raise SystemExit(f"{TRAIN_RECORDS} not found. Run scripts/phase1_prepare_data.py first.")
    records = load_records(TRAIN_RECORDS, args.limit)
    print(f"  {len(records):,} training records")

    # ----------------------------------------------------------------- model
    section("2. Load the 4-bit base model")
    model, tokenizer = load_model_and_tokenizer(model_config)
    # Batched *generation* needs left padding, but training needs right
    # padding -- left padding during training would place pad tokens between
    # the prompt and the labels. load_model_and_tokenizer sets left for the
    # eval path, so it is corrected here.
    tokenizer.padding_side = "right"
    print(f"  loaded {model_config['model']['id']}")

    section("3. Check sequence lengths against max_length")
    stats = token_length_stats(tokenizer, records[: min(len(records), 1000)])
    print(f"  tokens: min={stats['min']} median={stats['median']} p95={stats['p95']} max={stats['max']}")
    if stats["max"] > settings["max_length"]:
        over = stats["max"] - settings["max_length"]
        print(
            f"  WARNING: longest example exceeds max_length by {over} tokens and "
            "will be truncated, cutting the tail off its gold SQL."
        )

    # ------------------------------------------------------------------ lora
    section("4. Attach the LoRA adapter")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=settings["gradient_checkpointing"]
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    peft_config = LoraConfig(**lora_settings)
    print(f"  rank={lora_settings['r']} alpha={lora_settings['lora_alpha']}")
    print(f"  targets={lora_settings['target_modules']}")

    # --------------------------------------------------------------- trainer
    section("5. Train")
    dataset = build_sft_dataset(records)

    # Ensure the run logs enough points to assess a loss trend. A short smoke
    # run at the default logging_steps produces a single point, which makes
    # the Phase 3 gate vacuous -- it once reported FAIL for a run that had
    # trained perfectly well.
    effective_batch = (
        settings["per_device_train_batch_size"] * settings["gradient_accumulation_steps"]
    )
    optimizer_steps = max(
        1, int(len(records) * settings["num_train_epochs"] / effective_batch)
    )
    logging_steps = settings["logging_steps"]
    if args.logging_steps is not None:
        logging_steps = args.logging_steps
    elif optimizer_steps < logging_steps * 8:
        logging_steps = max(1, optimizer_steps // 8)
        print(
            f"  {optimizer_steps} optimiser steps expected -- lowering "
            f"logging_steps {settings['logging_steps']} -> {logging_steps} so the "
            "loss curve is measurable"
        )
    settings["logging_steps"] = logging_steps
    print(f"  effective batch {effective_batch}, ~{optimizer_steps} optimiser steps")

    sft_config = SFTConfig(
        output_dir=str(REPO_ROOT / "models" / "_checkpoints"),
        num_train_epochs=settings["num_train_epochs"],
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        gradient_checkpointing=settings["gradient_checkpointing"],
        optim=settings["optim"],
        learning_rate=settings["learning_rate"],
        lr_scheduler_type=settings["lr_scheduler_type"],
        warmup_ratio=settings["warmup_ratio"],
        max_grad_norm=settings["max_grad_norm"],
        weight_decay=settings["weight_decay"],
        max_length=settings["max_length"],
        logging_steps=settings["logging_steps"],
        save_strategy=settings["save_strategy"],
        seed=settings["seed"],
        packing=settings["packing"],
        bf16=torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.get_device_capability()[0] < 8,
        report_to=[],
        # Prompt-completion datasets default to completion-only loss; stated
        # explicitly so the behaviour is visible rather than inherited.
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"  trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train()
    elapsed = round(time.perf_counter() - started, 1)
    peak_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)

    history = [h for h in trainer.state.log_history if "loss" in h]
    losses = [h["loss"] for h in history]
    print(f"\n  trained in {elapsed}s, peak VRAM {peak_gib} GiB")
    if losses:
        print(f"  loss: first={losses[0]:.4f}  last={losses[-1]:.4f}")

    # ------------------------------------------------------------------ save
    section("6. Save the adapter")
    output_dir = Path(args.output)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  saved to {output_dir}")

    # ------------------------------------------------------------ smoke test
    section("7. Smoke test on held-out dev examples")
    model.config.use_cache = True
    trainer.model.eval()
    holdout = build_holdout(train_config["evaluation"]["smoke_test_size"], settings["seed"])
    smoke = run_smoke_test(trainer.model, tokenizer, holdout)

    # --------------------------------------------------------------- verdict
    loss_decreased, trend_detail = assess_loss_trend(losses)
    valid_sql = sum(bool(r["executed"]) for r in smoke)
    # `loss_decreased is None` means the run was too short to judge -- reported
    # as INDETERMINATE rather than silently counted as either outcome.
    passed = loss_decreased is True and valid_sql == len(smoke)
    verdict = (
        "INDETERMINATE" if loss_decreased is None else ("PASS" if passed else "FAIL")
    )

    from lora_text_to_sql.provenance import capture as capture_provenance

    report = {
        "phase": 3,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_result": verdict,
        "provenance": capture_provenance(REPO_ROOT),
        "model": model_config["model"]["id"],
        "lora": lora_settings,
        "training": settings,
        "n_train_records": len(records),
        "token_length_stats": stats,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": round(100 * trainable / total, 4),
        "train_runtime_seconds": elapsed,
        "peak_vram_gib": peak_gib,
        "final_train_loss": train_result.training_loss,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_decreased": loss_decreased,
        "loss_trend_detail": trend_detail,
        "optimizer_steps": optimizer_steps,
        "logged_loss_points": len(losses),
        "loss_history": history,
        "smoke_test": {
            "n": len(smoke),
            "produced_executable_sql": valid_sql,
            "execution_matches": sum(bool(r["execution_match"]) for r in smoke),
            "examples": smoke,
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"  loss decreased          {loss_decreased}  ({trend_detail})")
    print(f"  executable SQL          {valid_sql}/{len(smoke)} held-out examples")
    print(f"PHASE 3: {verdict}")
    print(f"Report written to {report_path}")
    print("=" * 72)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
