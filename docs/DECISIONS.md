# Decision log

Design decisions with their rationale, recorded as they are made. Anything
that turns out to be a constraint rather than a free choice is written down as
a constraint, including the ones that reduce the project's scope.

Status values: **Decided** (settled, with reasoning) · **Pending** (waiting on
a measurement) · **Superseded**.

---

## D-001: QLoRA on a 1.5B model, not full fine-tuning

**Status:** Decided · **Phase:** 0

The available GPU is an RTX 3050 Laptop with 4 GB of VRAM. Full fine-tuning of
even a 1.5B model needs roughly 24 GB once fp16 weights, gradients and AdamW
optimiser states are counted (~2 + 2 + 8 bytes per parameter, before
activations). That is ~6x the available budget, so full fine-tuning is not a
preference here, it is arithmetically excluded.

QLoRA resolves this on both axes: the base weights are frozen and quantised to
4-bit NF4 (~0.5 bytes/param), and gradients plus optimiser states exist only
for the LoRA adapter, typically well under 1% of parameters.

**Consequence:** the project demonstrates parameter-efficient fine-tuning
methodology. It cannot make claims about full fine-tuning, and does not.

---

## D-002: Qwen2.5-1.5B-Instruct as the base model

**Status:** Decided · **Phase:** 0

Chosen over similarly sized alternatives (TinyLlama-1.1B, Phi-3-mini,
Llama-3.2-1B) because it is the strongest 1.5B-class instruct model with a
permissive licence (Apache 2.0) and native `transformers` support, requiring
no `trust_remote_code`. Its instruct tuning matters for the baseline: the
before/after comparison is only interesting if the "before" model is a
credible zero-shot attempt, not a base model that cannot follow an
instruction at all.

**Fallback:** `Qwen2.5-0.5B-Instruct`, if Phase 0 or Phase 3 shows 1.5B does
not fit. Recorded in `configs/model.yaml` as `fallback_id`.

---

## D-003: Pin `device_map={"": 0}` rather than `"auto"`

**Status:** Decided · **Phase:** 0

`device_map="auto"` lets accelerate offload layers to CPU RAM when VRAM runs
short. In a script whose entire purpose is to answer "does this fit in 4 GB?",
that converts a genuine failure into a false pass.

Pinning every module to GPU 0 means the load either fits or raises OOM. The
Phase 0 script additionally walks every parameter asserting it is resident on
CUDA, so an offload cannot pass unnoticed.

**Consequence:** Phase 0 can fail where a more permissive config would appear
to succeed. That is the intent.

---

## D-004: Force a real bitsandbytes NF4 kernel round-trip, not just an import

**Status:** Decided · **Phase:** 0

The build plan flagged flaky native-Windows bitsandbytes support as the top
platform risk. `import bitsandbytes` succeeding does not establish that the
compiled CUDA backend works: the failure typically surfaces later, at the
first quantised matmul, after a multi-GB model download.

Phase 0 therefore quantises and dequantises a small tensor on the GPU and
checks the reconstruction error before any model is fetched. The failure
arrives in seconds instead of minutes, and it arrives with an unambiguous
cause.

---

## D-005: Exact version pins, torch installed separately

**Status:** Decided · **Phase:** 0

The project's central claim is a before/after comparison. If `transformers` or
`peft` changed between the baseline run and the post-fine-tuning run, a
metric shift could not be attributed to fine-tuning. All versions are pinned
exactly in `pyproject.toml` (originally `requirements.txt`, replaced when the
project became an installable package -- see D-028).

torch is excluded from that file on purpose: installing it from PyPI yields
the CPU-only wheel, which cannot run bitsandbytes 4-bit kernels. It has to
come from the CUDA wheel index, so it is a documented ordered step in
`docs/SETUP.md` rather than a line in a dependency list that would silently
install the wrong build.

Selected versions (latest available at project start, July 2026):
`transformers 5.14.1`, `peft 0.20.0`, `trl 1.9.2`, `bitsandbytes 0.50.0`,
`accelerate 1.14.0`, `datasets 5.0.1`, torch `2.13.0+cu126`.

**Note:** `transformers` 5.x deprecates the `torch_dtype` argument to
`from_pretrained` in favour of `dtype`. The Phase 0 script uses `dtype`.
Older QLoRA tutorials use `torch_dtype` and will emit deprecation warnings
against this pin.

---

## D-006: Python 3.10, acknowledged as at end-of-support-window

**Status:** Decided · **Phase:** 0

3.10 matches the known-working setup on the sibling projects in
`C:\Projects`, and newer Python installs on this machine have lacked wheel
support for parts of this stack. Every dependency above still publishes 3.10
wheels.

**Consequence, stated rather than hidden:** 3.10 is now the *minimum* supported
version across this entire stack, not a comfortable middle. The next major
release of any of these libraries is likely to drop it. This is known
maintenance debt, accepted deliberately for a time-boxed portfolio project.

---

## D-007: Training-memory probe is not part of the Phase 0 gate

**Status:** Decided · **Phase:** 0

A no-grad forward pass fitting in VRAM does **not** prove that training fits.
Backward passes add activation storage, gradients for the adapter, and
optimiser state. Treating an inference fit as a training green light would be
an overclaim.

Phase 0's pass criterion is therefore inference-only, matching the build
plan's wording ("load in 4-bit and run one forward pass"). The optional
`--train-probe` flag runs one LoRA forward+backward and reports peak memory as
an early indicator, explicitly labelled indicative rather than authoritative.
Phase 3 is where training feasibility is genuinely settled.

---

## D-008: Take WikiSQL from the canonical release, not a Hugging Face mirror

**Status:** Decided · **Phase:** 1

`load_dataset("wikisql")` does not work: `datasets` >= 3 no longer executes
dataset loading scripts, and `Salesforce/wikisql` contains only `wikisql.py`.

Every parquet mirror on the Hub was checked and rejected, because each is a
preprocessed derivative that has dropped the data this project depends on:

| Mirror | Problem |
|--------|---------|
| `htriedman/wikisql` | Instruction-tuned triples; SQL reads `FROM table`, no schema, no rows |
| `mlx-community/wikisql` | Flattened to a single `text` field; no table rows |
| `Rathan/wikisql` | Missing `train.tables.jsonl` and `test.tables.jsonl` |

Without table rows there is nothing to execute against, and execution match
is the entire premise. The project therefore downloads
`https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2`, the same
archive the official Hugging Face loading script fetches (BSD-3).

Verified after download: 56,355 / 8,421 / 15,878 examples, matching the
published split sizes exactly. The script asserts these counts, so a
truncated or substituted download fails loudly instead of quietly training on
less data.

---

## D-009: Rebuild the SQLite tables instead of using the shipped `.db` files

**Status:** Decided · **Phase:** 1

The release ships `train.db` / `dev.db` / `test.db`. They cannot be used here:

```sql
CREATE TABLE table_1_10015132_11 (col0 text, col1 text, col2 text, ...)
-- values also stored lowercased: 'antonio lang', 'duke'
```

Columns are anonymised to `col0..colN`. Prompting a model with those reduces
the task to guessing a column *index*, which is not the task this project
measures: generating SQL against real, human-readable schemas. Tables are
therefore rebuilt from `*.tables.jsonl`, preserving original headers and
value casing.

Rebuilding also forces the schema decisions below to be made explicitly
rather than inherited.

---

## D-010: Declare text columns `COLLATE NOCASE`

**Status:** Decided · **Phase:** 1 · **Highest-impact decision in Phase 1**

WikiSQL condition values are extracted from the question text, so their casing
routinely differs from the stored cell: `'russia'` vs `Russia`,
`'louisville'` vs `Louisville`. Measured by executing the **ground-truth** SQL
over a random sample of 500 dev examples:

| Column declaration | Ground truth returns a usable result |
|--------------------|--------------------------------------|
| `TEXT` (case-sensitive) | 276 / 500, **55.2%** |
| `TEXT COLLATE NOCASE`   | 472 / 500, **94.4%** |

Case sensitivity alone accounts for ~39 percentage points. Left unaddressed,
roughly 45% of the reference queries return nothing, and both the baseline and
the fine-tuned model would be scored against a reference that is broken almost
half the time, making the headline before/after comparison meaningless.

The official WikiSQL evaluation solves this by lowercasing every stored value.
Declaring the collation achieves the same matching behaviour while keeping the
original casing visible in the prompt, which is what a model would encounter
against a real database.

**Limitation, recorded not hidden:** SQLite's built-in `NOCASE` folds ASCII
A–Z only. Case differences in non-ASCII text (`Cerro Porteño`) are not folded.

---

## D-011: Type numeric columns as `REAL`, not `TEXT`

**Status:** Decided · **Phase:** 1

WikiSQL marks each column `text` or `real`, and 9,562 `>` plus 9,030 `<`
conditions across the corpus depend on that typing. Stored as text, SQLite
compares lexicographically: `'9' > '10'` is **true**, silently corrupting
every range query and every `SUM`/`AVG`.

Values in `real` columns are coerced with `float()` after stripping thousands
separators (`"48,065"` → `48065.0`, which bare `float()` rejects).

Measured before committing to this: **0 of 4,306** `real` columns in the dev
split contain a non-numeric value, so the coercion is safe rather than
lossy. A regression test pins the behaviour.

---

## D-012: Exclude degenerate ground truth from the evaluation set

**Status:** Decided · **Phase:** 1

Roughly **8%** of examples have ground-truth SQL that returns nothing usable:
either an empty result set, or a single `NULL` from an aggregate over zero
matching rows. Measured across 2,000-example samples: 91.6% / 92.1% / 92.5%
usable for train / dev / test.

These must not appear in the evaluation set. Scoring compares result sets, so
if the reference answer is "nothing", then *any* query returning nothing
(including syntactically valid nonsense selecting an unrelated column) scores
as correct. Keeping them would inflate both the baseline and the fine-tuned
number by rewarding failure, and would do so unevenly between the two.

The eval set is therefore drawn only from examples whose ground truth returns
a non-degenerate result. 500 such examples were drawn from 550 inspected.

---

## D-013: Exclude four tables with case-colliding headers

**Status:** Decided · **Phase:** 1

SQLite compares column names case-insensitively, so a table declaring both
`% Total budget` and `% total budget` fails at `CREATE TABLE`. The corpus has
**no exact duplicate headers**, but four tables (3 train, 1 test) collide once
case is folded.

They are excluded rather than repaired. Repairing means renaming a column,
which breaks the invariant that a column can be referenced by its real header
name: the annotation selects columns by *index*, so a renamed column would
silently resolve to the wrong one.

Exclusion is lossless here: **zero examples in any split reference these four
tables.** The script asserts that rather than assuming it, and fails loudly if
a future corpus revision makes the exclusion lossy.

---

## D-014: One database per split

**Status:** Decided · **Phase:** 1

The build plan sketches `data/tables/*.db`. Implemented as one database per
split (`train.db`, `dev.db`, `test.db`) rather than one file per table: the
train split has 18,585 tables, and that many files would be slow to create and
awkward to move. Table names are unique within a split, so a single file is
unambiguous. Materialised sizes: 83.4 / 12.0 / 23.9 MiB.

---

## D-015: Stratify the training subsample by query shape

**Status:** Decided · **Phase:** 1

The 4 GB GPU makes the full 56K train split impractical in the time budget, so
training uses a 6,000-example subsample. Sampling is stratified by
`(aggregation type, condition count)`, the two properties that determine
query difficulty in WikiSQL.

Uniform random sampling would approximate the distribution anyway; stratifying
makes it exact and guarantees the rare shapes (`SUM`/`AVG` with 3–4
conditions) actually appear rather than being left to chance. Largest-remainder
allocation keeps the total exact.

Achieved maximum per-stratum drift versus the full train split: **0.0001**.

---

## D-016: One evaluation script, not one per phase

**Status:** Decided · **Phase:** 2

The build plan's central claim is that identical eval code scores the
baseline and the fine-tuned model. The most reliable way to guarantee that is
for only one scoring program to exist, so there is no `phase2_eval.py` and no
`phase4_eval.py`: only `scripts/run_eval.py`, and the sole difference between
the runs is a flag:

```bat
python scripts\run_eval.py --name baseline
python scripts\run_eval.py --name finetuned --adapter models\lora-adapter
```

Two scripts would be equivalent only for as long as nobody edited one of them.
This makes the guarantee structural rather than a matter of discipline.

---

## D-017: Extract SQL from model output before scoring

**Status:** Decided · **Phase:** 2

An instruction-tuned model told to emit "only SQL" still returns markdown
fences, a `SQL:` prefix, or a trailing sentence. Feeding raw output to SQLite
would score those as failures even when the query underneath is correct.

This is a **fairness** decision, and it cuts in the direction that matters:
the un-fine-tuned baseline is markedly chattier than a fine-tuned model will
be, so a naive harness would score the baseline near zero for formatting
reasons and manufacture an improvement that fine-tuning did not earn.
Extraction is applied identically to both runs.

Guards against over-extraction in the other direction: only the first
statement is taken (semicolon-aware, so a `;` inside a string literal does not
truncate a valid query), and non-read-only statements are rejected outright.

---

## D-018: Greedy decoding

**Status:** Decided · **Phase:** 2

`do_sample=False`, temperature unset. A before/after comparison that moves
when rerun is not a measurement. Greedy decoding makes both runs
deterministic and reproducible.

---

## D-019: Result-set comparison rules

**Status:** Decided · **Phase:** 2

Three choices, each of which changes the score:

- **Row order ignored.** No gold query contains `ORDER BY`, so row order has
  no meaning and SQLite does not guarantee it.
- **Column order significant.** Tuples are compared positionally: selecting
  different columns is a different answer.
- **Multiplicity preserved.** Results are compared as sorted lists, not sets:
  returning a row twice is not the same answer as returning it once.

Values are normalised before comparison: ints and floats unify (`COUNT`
returns `12`, a REAL column returns `12.0`, the same answer), floats compare
within `1e-6` since `SUM`/`AVG` accumulate representation error, and strings
are casefolded. Casefolding cannot let a wrong query pass, because result
casing is determined by the stored data rather than by the model.

---

## D-020: Guard against SQLite's double-quoted-string misfeature

**Status:** Decided · **Phase:** 2 · **Caught by the harness self-test**

For MySQL compatibility, SQLite silently reinterprets a double-quoted
identifier as a **string literal** when it does not name a real column:

```
SELECT "Nope" FROM t   ->  [('Nope',), ('Nope',)]        no error
SELECT [Nope] FROM t   ->  OperationalError: no such column
SELECT  Nope  FROM t   ->  OperationalError: no such column
```

Only the double-quoted form fails silently, and the prompt presents columns
double-quoted, so that is the *likely* hallucination shape, not an exotic one.

Measured on the 500-example evaluation set, a prediction that selects a
plausible but entirely non-existent column:

| | syntactic validity | execution accuracy |
|---|---:|---:|
| Without the guard | **100.00%** | 0.00% |
| With the guard | 0.00% | 0.00% |

Uncaught, this would have reported a perfect validity rate for queries that
are wrong in every case, and "validity improved" is exactly the kind of claim
a fine-tuning write-up leans on.

SQLite can disable the behaviour via `SQLITE_DBCONFIG_DQS_DML`, but Python
only exposes `Connection.setconfig` from 3.12 and this project targets 3.10
(D-006), so predicted SQL is instead scanned for double-quoted identifiers
that do not name a real column. Identifiers introduced by `AS` are allowed,
since an alias is legitimate and does not change the result set.

---

## D-021: The harness self-tests against known-answer inputs

**Status:** Decided · **Phase:** 2

A scoring harness that is never itself tested is the easiest way to produce a
confident and wrong before/after number. `run_eval.py --self-test` runs the
real 500-example evaluation set through strategies whose correct score is
known in advance, and needs no GPU:

| Strategy | Metric | Observed | Expected |
|---|---|---:|---|
| Gold SQL verbatim | execution | 1.0000 | ~1.0 |
| Gold SQL restyled (`[col]`, lowercase keywords) | execution | 1.0000 | ~1.0 |
| Gold SQL restyled | exact match | 0.0000 | ~0.0 |
| Gold SQL wrapped in markdown + prose | execution | 1.0000 | ~1.0 |
| A different column from the same table | execution | 0.0000 | ~0.0 |
| `WHERE` clause removed | execution | 0.0140 | low |
| Hallucinated column | validity | 0.0000 | 0.0 |
| Non-SQL prose | validity | 0.0000 | 0.0 |

Rows 2 and 3 are the project's premise stated as a measurement: **the same
queries score 100% on execution match and 0% on exact string match.** String
similarity would have scored 500 semantically perfect queries at zero.

Row 6 is not zero because for ~1.4% of examples the `WHERE` clause does not
change the result: the condition already matches every row. That is a
property of the data, not a harness defect.

Writing this self-test is what surfaced D-020, and also surfaced that 117 of
26,531 tables have headers containing square brackets (`Kickoff [a ]`), which
makes `[...]` quoting unsafe for them.

---

## D-022: TRL `SFTTrainer` over a hand-written training loop

**Status:** Decided · **Phase:** 3

The build plan allowed either. TRL was chosen after reading the pinned
version's source rather than assuming its API, because `trl` 1.9.2 postdates
several breaking renames and a guessed API is exactly what broke Phase 0.

Two findings decided it:

- `SFTConfig` exposes `completion_only_loss`, which defaults to `True` for
  prompt-completion datasets. Masking the prompt out of the loss is the single
  most important training detail here (D-023), and getting it natively is
  better than reimplementing label masking by hand.
- `SFTTrainer` accepts `peft_config` directly, so adapter attachment,
  gradient checkpointing and the training loop stay in one well-exercised
  code path rather than three hand-rolled ones.

Every argument passed to `SFTConfig`, `LoraConfig` and
`prepare_model_for_kbit_training` was checked against the installed source by
AST inspection before being used. `task_type` is not declared on `LoraConfig`
itself; it is inherited from `PeftConfig`, which a shallower check would
have flagged as a false error.

---

## D-023: Compute loss on the completion only

**Status:** Decided · **Phase:** 3

Loss is computed on the assistant turn (the SQL), not on the prompt. Training
on prompt tokens would spend model capacity memorising table schemas that are
supplied at inference time anyway, and would dilute the gradient signal for
the only thing being learned: the question-to-SQL mapping.

Implemented via the TRL *conversational prompt-completion* dataset type
(`{"prompt": [...messages], "completion": [assistant message]}`) with
`completion_only_loss=True` set explicitly rather than left to default, so the
behaviour is visible in the config instead of inherited silently.

---

## D-024: Training and evaluation share one prompt builder

**Status:** Decided · **Phase:** 3 · **Quietest failure mode in the project**

`dataset.to_sft_example()` and the Phase 2/4 eval harness both call
`prompt.build_messages()`. There is one function and two callers.

This matters more than it looks. If the model were fine-tuned on a prompt
format that differed from the evaluation format (by a system-prompt wording,
a stray newline, a different schema rendering), then the Phase 4 number would
be a mixture of genuine learning and format mismatch, in an unknown
proportion and an unknown direction. Nothing in the output would look wrong;
the number would simply be untrue.

`tests/test_dataset.py` asserts the eval prompt is byte-identical to the
training prompt, and that the gold SQL never leaks into the prompt side.

---

## D-025: Memory settings sized to the *measured* budget

**Status:** Decided · **Phase:** 3

Phase 0 measured 3.228 GiB free of 4.0 GiB total: roughly 770 MB is consumed
by the CUDA context, the Windows WDDM reserve and the desktop compositor
before any model loads. The training budget is therefore ~3.2 GiB, not 4.0,
and the settings follow from that:

| Setting | Value | Reason |
|---|---|---|
| `per_device_train_batch_size` | 1 | Forced by VRAM |
| `gradient_accumulation_steps` | 16 | Restores an effective batch of 16 |
| `gradient_checkpointing` | true | Recomputes activations instead of storing them; ~30% slower, not optional |
| `optim` | `paged_adamw_8bit` | Small optimiser state, and pages to host RAM under pressure rather than raising OOM |
| `packing` | false | Would let one example's SQL condition the next, and complicates completion-only loss accounting |

Fallback order if the Phase 0 `--train-probe` peak leaves too little headroom:
rank 16 → 8, then drop the MLP projections from `target_modules`, then shorten
`max_length`, then the 0.5B model. Exposed as `--rank` and `--attention-only`
flags so the fallback needs no code edit.

---

## D-026: Sequence lengths are checked before training, not after

**Status:** Decided · **Phase:** 3

`max_length` truncation is silent. A gold SQL statement cut off at the tail
would train the model toward incomplete queries, and the symptom (a mediocre
Phase 4 execution accuracy) gives no hint of the cause. Phase 3 therefore
measures the token-length distribution against `max_length` first and warns
before a single step runs.

---

## D-027: Held-out smoke examples come from dev, not train or test

**Status:** Decided · **Phase:** 3

The Phase 3 gate ("valid SQL on 3 held-out examples") draws from the **dev**
split. Drawing from the train subsample would measure memorisation. Drawing
from the Phase 2/4 test set would leak the evaluation set into a decision made
during training, which is precisely the contamination the official-splits
decision (D-002 era) was meant to avoid.

---

## D-028: Package as an installable project, not `sys.path` hacks

**Status:** Decided · **Phase:** repo hygiene, no phase number

Every script and test file inserted `sys.path.insert(0, str(REPO_ROOT /
"src"))` before importing from `lora_text_to_sql`, because `src/` was never
an installed package -- eleven near-identical lines whose only job was
working around that, each paired with a `# noqa: E402` to silence the
import-order lint it created. `pip install -e .`, via a new
`pyproject.toml`, makes the package importable from anywhere without the
hack, so all eleven lines and their noqa comments were removed.

torch is left out of `pyproject.toml`'s dependency list entirely, which is
not quite what `requirements.txt` did: that file pinned
`torch==2.13.0+cu130` inline behind an `--extra-index-url` line, a trick
specific to a plain requirements file. `pyproject.toml`'s standard
`dependencies` array has no equivalent per-package index, so a plain
`torch` entry there would risk resolving the CPU wheel from PyPI silently --
the exact failure D-005 already documents. torch stays a separately
installed step (`docs/SETUP.md` step 3, run before `pip install -e .`),
which is what the setup instructions already required in practice.
`requirements.txt` itself is removed -- `pyproject.toml` is now the single
source of pinned versions, and two files pinning the same versions is
exactly the kind of duplication this project's own `io.py` consolidation
argues against elsewhere.

---

## Resolved: answered by the runs

- **P-001: Does bitsandbytes 4-bit work on native Windows?** **Yes.**
  bitsandbytes 0.50.0 ships an official `win_amd64` wheel and the NF4
  quantise/dequantise round-trip succeeded on GPU (mean absolute
  reconstruction error 0.073, the expected magnitude for NF4 on
  standard-normal input). **The WSL2 fallback was not needed** and remains
  documented in `docs/SETUP.md` only as a contingency.

- **P-002: Does Qwen2.5-1.5B-Instruct fit in 4-bit within 4 GB?** **Yes,
  comfortably.** 4-bit weights occupy 1.119 GiB; after a forward pass, 2.066
  GiB of 4.0 GiB is in use with 1.933 GiB free, and no parameter was offloaded
  to CPU. Note the usable budget is ~3.23 GiB rather than 4.0: the CUDA
  context and Windows desktop compositor consume ~770 MB before the model
  loads.

- **P-003: Is 1.5B viable for training, or is the 0.5B fallback required?**
  **1.5B is viable at full rank.** Training peaked at 2.628 GiB with rank 16
  across all seven projection modules. **The 0.5B fallback was not needed**,
  and neither were the `--rank 8` or `--attention-only` reductions.

- **P-004: Pin `model.revision` to a commit SHA.** **Done.** Pinned to
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The repository's last
  modification is 2024-09-25, well before this project's runs, so that SHA is
  provably the revision the reported results were produced with.

  Pinning it in the config was not sufficient on its own: `revision` was never
  passed to `from_pretrained`, so the pin was inert and both loaders silently
  defaulted to `main`. It is now threaded through
  `generation.load_model_and_tokenizer` and `phase0_verify_env`. A pinned
  version that nothing reads is worse than no pin, because it looks like a
  guarantee.

- **P-005: Close the O-001 epoch confound.** **Done.** Re-ran n=200 at 2
  epochs (was 1) via `run_scaling_study.py --sizes 200 1000`, and added an
  n=1,000 point that did not exist before, locating the knee rather than
  interpolating it. Corrected curve: n=200 now captures 82% of the total
  gain (previously reported as 79% under the mismatched epoch count),
  n=1,000 captures 92%, n=6,000 captures 100%. The direction did not change;
  the exact share did, which is exactly the risk O-001 flagged.

---

## Open: deliberately not done

- **O-002: No dev-set evaluation during training.** Phase 3's gate is a
  3-example smoke test, which is too small to detect overfitting. Two epochs
  over 6,000 examples on a 1.5B model is unlikely to overfit badly, and the
  held-out test result (85.4%) is evidence it did not, but a per-epoch dev
  evaluation would have made that an observation rather than an inference.

- **O-003: Single-table only.** WikiSQL has no joins. Spider is the natural
  next step and is a substantially harder task. Scoped as a separate project
  in [issue #6](https://github.com/chinmaynighojkar/wikisql-qlora-execution-eval/issues/6)
  rather than attempted here -- it needs a new data loader, a multi-table
  identifier guard, and a re-verified VRAM budget for longer prompts, not a
  quick extension of this pipeline.
