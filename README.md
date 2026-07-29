# QLoRA Text-to-SQL

QLoRA fine-tuning of **Qwen2.5-1.5B-Instruct** for natural-language-to-SQL,
evaluated by **execution match** — generated SQL is run against real SQLite
tables and its result set compared to ground truth, not scored on string
similarity.

Trained entirely on a **4 GB RTX 3050 Laptop GPU**.

| Metric | Base model | + QLoRA adapter | Change |
|---|---:|---:|---:|
| **Execution accuracy** (primary) | 39.8% | **85.4%** | **+45.6 pts** |
| Syntactic validity | 55.2% | 99.8% | +44.6 pts |
| Exact string match (secondary) | 3.4% | 77.8% | +74.4 pts |

500 held-out WikiSQL test examples. Paired McNemar test: 238 fixed, 10 broken,
χ² = 207.8, **p < 1e-15**. 95% Wilson intervals [35.6%, 44.2%] vs
[82.0%, 88.2%] — non-overlapping.

---

## The problem

A small open model asked to write SQL against an unfamiliar table schema fails
most of the time — and it fails in a specific, uninteresting way. Here is the
un-fine-tuned model on three real test questions:

```sql
-- base model                                    -- what SQLite says
SELECT Original_air_date FROM t WHERE Season_# = 4    unrecognized token: "#"
SELECT Oricon Albums Chart FROM t WHERE ...           near "Chart": syntax error
SELECT MIN(Crowd) FROM t WHERE Away_team = '...'      no such column: Away_team
```

It understands the *question* well enough — it picks plausible columns — but it
cannot quote an identifier. WikiSQL column headers routinely contain spaces,
slashes and punctuation (`Away team`, `Season #`, `School/Club Team`,
`Time/Retired`), and **44.8% of baseline generations never executed at all**.

## The solution

QLoRA fine-tuning: freeze the base model in 4-bit NF4, train only a rank-16
LoRA adapter — 18.5M parameters, 2.04% of the model — on 6,000 WikiSQL
examples. On the same three questions afterwards:

```sql
SELECT "Original air date" FROM "table_15824796_5" WHERE "Season #" = 4
SELECT "Oricon Albums Chart" FROM "table_23180638_1" WHERE "Debut Sales (copies)" > 339333.011497678
SELECT MIN("Crowd") FROM "table_2_10640687_18" WHERE "Away team" = 'footscray'
```

All three correct. `sql_error` failures fell **219 → 0**.

## The outcome

Execution accuracy **39.8% → 85.4%**, verified significant and leakage-free
(see [Is it real?](#is-it-real)). Training took **112 minutes** and peaked at
**2.63 GiB** of VRAM.

### The more useful finding

Measuring more than one training size turned a before/after into a
data-efficiency curve:

| Train examples | Execution accuracy | Gain | Share of total gain | Train time |
|---:|---:|---:|---:|---:|
| 0 | 39.8% | — | 0% | — |
| 200 | 76.0% | +36.2 pts | **79%** | **102 s** |
| 6,000 | 85.4% | +45.6 pts | 100% | 112 min |

**200 examples and 102 seconds of training captured 79% of the total
achievable gain.** The remaining 21% cost 30× the data and 66× the training
time — a 249× worse marginal return per second of compute.

For anyone budgeting a fine-tune, that is the more actionable result: the
question is not "does fine-tuning help" but "how little is enough", and here
the answer was a couple of hundred examples.

> **Caveat, stated plainly:** the n=200 point ran 1 epoch and the n=6,000 point
> ran 2, so that row varies data *and* epochs together. The direction is not in
> doubt — 36.2 of 45.6 points from 1.5% of the training time — but the exact
> share would shift under a clean 2-epoch run at n=200. Re-running is a
> 3-minute job (`run_scaling_study.py --sizes 200 1000`) and would also locate
> the knee, which is currently interpolated.

---

## Is it real?

A +45-point swing is exactly when a result deserves the most scrutiny, so the
comparison ships with its own checks rather than a bare delta.

**The improvement is statistically significant.** Both models were scored on
the same 500 examples, so the outcomes are paired and McNemar's test is the
correct one — it uses only the discordant pairs, the examples where the two
models disagree.

| | Fine-tuned correct | Fine-tuned wrong |
|---|---:|---:|
| **Baseline correct** | 189 | 10 |
| **Baseline wrong** | 238 | 63 |

χ² = 207.8 (continuity corrected), p < 1e-15.

**There is no train/test leakage.** 0 tables are shared between the 6,000
training examples and the 500 evaluation examples. One question *string*
appears in both ("Which country has a rank of 5?") but against different
tables with different gold SQL. WikiSQL's official splits have disjoint
tables; this verifies that rather than citing it.

**The gain is not only from queries becoming runnable.** If every point came
from invalid SQL becoming valid, the model would have learned syntax and
nothing about the question:

| | Executed | Correct | Accuracy **given** execution |
|---|---:|---:|---:|
| Base model | 276 | 199 | 72.1% |
| + adapter | 499 | 427 | 85.6% |

Both effects are present: far more queries run, *and* those that run are more
often right.

**The scoring harness is itself tested.** `run_eval.py --self-test` scores the
real evaluation set with inputs whose correct result is known in advance —
gold SQL verbatim, gold SQL restyled, gold SQL buried in markdown, a wrong
column, a hallucinated column, non-SQL prose. A harness that is never itself
tested is the easiest way to produce a confident and wrong number.

**Regressions are reported, not just fixes.** Fine-tuning broke 10 examples it
had previously got right. The most interesting is a genuine ambiguity:

```sql
-- Q: "How many points total are there later than 2003?"
-- gold:      SELECT SUM("Points")   FROM t WHERE "Year" > 2003
-- base:      SELECT SUM(Points)     FROM t WHERE Year > 2003     correct
-- + adapter: SELECT COUNT("Points") FROM t WHERE "Year" > 2003   wrong
```

"How many … total" supports both readings. The adapter latched onto "how
many" → `COUNT`; the gold annotation reads "total" → `SUM`.

---

## Why execution match, not string match

Two syntactically different queries can be semantically identical — column
ordering, aliasing, quoting style, whitespace. String comparison scores
correct answers as wrong.

This is not a theoretical concern. On the real baseline generations, execution
match scored **39.8%** where exact string match scored **3.4%** — an 11.7×
difference. A string-similarity harness would have reported this model as
almost totally useless and the experiment would have ended there.

The harness runs each generated query against the materialised SQLite table
and compares **result sets**, order-insensitively (no gold query contains
`ORDER BY`, so row order is meaningless), with column order significant and
row multiplicity preserved. This is the methodology the WikiSQL and Spider
leaderboards use, so it is defensible as standard practice rather than
invented here.

> Exact string match is reported as a **secondary** metric only. Its jump to
> 77.8% partly reflects the model learning this project's specific gold-SQL
> rendering convention, not just better SQL. Execution accuracy is the
> defensible number.

---

## Three bugs that would have produced a wrong headline

Each was caught by a check written before the result existed. They are the
substance of the project.

**1. Case sensitivity would have broken the reference itself.** WikiSQL
condition values come from the question text, so their casing differs from the
stored cell (`'russia'` vs `Russia`). Executing the *ground-truth* SQL against
case-sensitive columns returned a usable result for only **55.2%** of
examples; with `COLLATE NOCASE`, **94.4%**. Left unfixed, both models would
have been scored against a reference that was broken almost half the time.

**2. SQLite silently turns unknown double-quoted identifiers into string
literals.** A MySQL-compatibility misfeature:

```
SELECT "Nope" FROM t   ->  [('Nope',), ('Nope',)]   no error
SELECT [Nope] FROM t   ->  OperationalError: no such column
```

Only the double-quoted form fails silently — and the prompt presents columns
double-quoted, so it is the *likely* hallucination shape. Measured: a
prediction selecting a wholly non-existent column scored **100% syntactic
validity at 0% accuracy** without a guard.

The guard needed refining once the baseline's real output arrived. The base
model writes `WHERE No. = "21"` — a misquoted *value*, not a hallucinated
column. SQLite executes it and returns the right rows, so rejecting it would
have penalised the baseline for a quoting habit and inflated the fine-tuned
model's apparent gain. The check now distinguishes identifier position from
value position.

**3. ~8% of examples have degenerate ground truth** — an empty result set, or
`NULL` from an aggregate over zero matching rows. When the correct answer is
"nothing", *any* query returning nothing scores as correct, including valid
nonsense. These are excluded from the evaluation set.

Full reasoning for all of these, and 24 other decisions, in
**[docs/DECISIONS.md](docs/DECISIONS.md)**.

---

## Hardware constraint

An **RTX 3050 Laptop, 4 GB VRAM** — of which only **3.23 GiB** is actually
free once the CUDA context and Windows desktop compositor are accounted for.

This is the binding constraint behind most of the design. Full fine-tuning of
even a 1.5B model needs roughly 24 GB once fp16 weights, gradients and AdamW
state are counted — about 6× the budget. QLoRA is not a preference here, it is
arithmetic.

| | |
|---|---:|
| 4-bit NF4 weights | 1.119 GiB |
| Peak during training | 2.628 GiB |
| Trainable parameters | 18,464,768 (2.04%) |
| Training time (6,000 × 2 epochs) | 112 min |
| Training loss | −72.0% (first vs last quartile) |

Settings that made it fit: batch size 1 with 16-step gradient accumulation,
gradient checkpointing, `paged_adamw_8bit`, and completion-only loss.

---

## Architecture

```
WikiSQL official release (BSD-3, train 56,355 / dev 8,421 / test 15,878)
      │
      ▼
[Table Materializer] ──► real SQLite tables, one DB per split, preserving
      │                  original headers and value casing, so SQL can be
      │                  EXECUTED. Ground truth verified by execution:
      │                  0 errors in 6,000 sampled queries.
      ▼
[Prompt Builder] ──────► one template, shared by training and evaluation
      │
      ▼
[Qwen2.5-1.5B-Instruct, 4-bit NF4]   same model both runs; only weights differ
      │
      ├─────────────► [Execution-Match Harness] ──► baseline_metrics.json
      │                                             (committed BEFORE training)
      ▼
[QLoRA SFT] ──► rank-16 LoRA on attention + MLP projections
      │
      ▼
base (frozen, 4-bit) + trained adapter
      │
      └─────────────► [Execution-Match Harness] ──► finetuned_metrics.json
                       ↑ identical code path        + comparison.md
```

**One evaluation script serves both runs.** There is deliberately no
`phase2_eval.py` and no `phase4_eval.py` — only `run_eval.py`, with
`--adapter` as the sole difference. Two scripts would stay identical only
until someone edited one.

**Training and evaluation share one prompt builder.** If the model were
fine-tuned on a prompt format differing from the evaluation format — by a
system-prompt wording, even a newline — the result would be a mixture of
learning and format mismatch, in unknown proportion. A test asserts the two
prompts are byte-identical.

## Tech stack

`transformers 5.14.1` · `peft 0.20.0` · `trl 1.9.2` · `bitsandbytes 0.50.0` ·
`accelerate 1.14.0` · `datasets 5.0.1` · `torch 2.13.0+cu130` ·
`sqlite3` (stdlib) · Python 3.10

## Build phases

Each phase had a verification gate; the next did not start until it passed.

| Phase | Gate | Result |
|---:|---|---|
| 0 | 4-bit load + forward pass without OOM | **PASS** — 1.119 GiB, no CPU offload |
| 1 | Tables materialised; ground-truth SQL executes | **PASS** — 0 errors in 6,000 |
| 2 | Baseline scored, `baseline_metrics.json` committed | **PASS** — 39.8% |
| 3 | Decreasing loss; valid SQL on 3 held-out examples | **PASS** — −72.0% loss, 3/3 executable |
| 4 | Identical harness, before/after comparison | **PASS** — +45.6 pts, p < 1e-15 |
| 5 | Write-up with real achieved numbers | This document |

**139 tests** (`pytest`), with regression guards on the collation,
numeric-typing, extraction, result-comparison, prompt-parity and loss-trend
decisions.

## Reproducing

```bat
python scripts\phase0_verify_env.py --train-probe    REM GPU gate
python scripts\phase1_prepare_data.py                REM CPU only
python -m pytest                                     REM CPU only
python scripts\run_eval.py --self-test               REM CPU only

python scripts\run_eval.py --name baseline
python scripts\phase3_train.py
python scripts\run_eval.py --name finetuned --adapter models\lora-adapter
python scripts\phase4_compare.py
```

Phase 1 is seeded (`SEED = 20260728`); repeated runs produce identical
subsamples. Decoding is greedy throughout — a before/after that moves when
rerun is not a measurement.

## Scope and limitations

Stated up front rather than discovered later.

- **Single-table only.** WikiSQL contains no joins. No claim is made about
  multi-table or general enterprise text-to-SQL, which is substantially
  harder.
- **Not production-ready.** This is a fine-tuning *methodology*
  demonstration — no serving layer, no handling of ambiguous questions, no
  guarantees against a model emitting destructive SQL beyond the harness's
  read-only guard.
- **The evaluation set excludes degenerate ground truth** (~8% of examples).
  This makes scoring meaningful but means the number is not directly
  comparable to leaderboard figures computed over the full test split.
- **`COLLATE NOCASE` folds ASCII only.** Case differences in non-ASCII text
  (`Cerro Porteño`) are not folded.
- **The data-efficiency curve has two points and an epoch confound** (above).
- **One model, one dataset.** No claim that the improvement generalises to
  other model families or SQL dialects.

## Repository layout

```
configs/     model.yaml (shared by every phase), training.yaml
scripts/     phase0_verify_env, phase1_prepare_data, run_eval,
             phase3_train, phase4_compare, run_scaling_study
src/         wikisql, materialize, prompt, dataset, extraction,
             evaluate, generation
tests/       139 tests
reports/     metrics, comparison.md, scaling_curve.md (committed)
docs/        SETUP.md, DECISIONS.md, RESUME_BULLETS.md
```

## Related work in this portfolio

Built to the same standard as
[`heart-disease-mlops`](https://github.com/chinmaynighojkar/heart-disease-mlops):
real measured numbers only, limitations documented rather than omitted, and
design decisions recorded with their reasoning. The two projects share no code
— the standard is the only thing in common.
