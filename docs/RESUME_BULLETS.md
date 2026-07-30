# Resume bullets, STAR story, and interview prep

Every number here traces to a committed artefact in `reports/`. Nothing is
rounded up, and the limitations are included because they are what make the
claims survive follow-up questions.

---

## Resume bullets

Pick 2–3. The first is the strongest general-purpose bullet; the second is
better for roles that emphasise evaluation or work under regulatory scrutiny.

**Primary — outcome-led**

> Fine-tuned Qwen2.5-1.5B for natural-language-to-SQL using QLoRA (4-bit NF4 +
> rank-16 LoRA) on a 4 GB laptop GPU, raising execution accuracy from 39.8% to
> **85.4%** on 500 held-out WikiSQL examples — validated with a paired McNemar
> test (238 fixed vs 10 broken, p < 1e-15) and explicit train/test leakage
> checks.

**Rigour-led — for ML Engineer / regulated employers**

> Built an execution-match evaluation harness that runs generated SQL against
> real SQLite tables and compares result sets, rather than scoring string
> similarity — a distinction worth **11.7×** on the same model output (39.8%
> execution accuracy vs 3.4% exact match). Self-tested the harness against
> known-answer inputs, which surfaced a SQLite double-quoted-identifier
> misfeature that would have reported **100% syntactic validity for queries
> that were wrong in every case**.

**Efficiency-led — for resource-constrained or startup roles**

> Measured the data-efficiency curve of QLoRA fine-tuning at three scales,
> all trained for the same 2 epochs so data size was the only variable: 200
> examples (12 minutes of training) captured **82% of the total achievable
> accuracy gain**, 1,000 examples captured 92%, and the full 6,000-example
> run bought only the last 8% for nearly 10× the training time of the 200
> example run.

**Engineering-detail bullet — pairs well with any of the above**

> Trained 18.5M adapter parameters (2.04% of the model) within a 3.2 GiB VRAM
> budget using gradient checkpointing, 16-step gradient accumulation, paged
> 8-bit AdamW and completion-only loss; documented 27 design decisions with
> their measured justification, backed by 139 tests.

**Process-led — for roles that value code review and reliability**

> Ran a five-axis code review (correctness, readability, architecture,
> security, performance) against the finished pipeline and found two silent
> failure modes the test suite had not caught: a scaling study step that
> could reuse a stale evaluation result after retraining, and a comparison
> script that divided by zero on a degenerate run. Fixed both, added a
> modification time check and a zero guard respectively, and confirmed all
> 139 tests still passed before merging.

---

## STAR story

**Situation.** My portfolio covered RAG, MLOps and classical ML but had no
fine-tuning, which was plausibly filtering me out of roles titled *ML
Engineer* rather than *AI Engineer*. The only hardware available was an RTX
3050 Laptop with 4 GB of VRAM — and only 3.23 GiB of that actually free once
the CUDA context and desktop compositor were counted.

**Task.** Demonstrate that fine-tuning measurably improves a small model at a
real task, with a before/after comparison that would survive a sceptical
interviewer. The measurement mattered more to me than the number: anyone can
claim a delta.

**Action.**

- Chose **QLoRA** — not as a preference but from arithmetic. Full fine-tuning
  of a 1.5B model needs ~24 GB once fp16 weights, gradients and AdamW state
  are counted, roughly 6× the budget. Freezing the base in 4-bit NF4 and
  training a rank-16 adapter brought the peak to 2.63 GiB.
- Built an **execution-match harness** instead of string comparison, because
  semantically identical SQL can differ textually. On real generations this was
  worth 11.7× — execution match 39.8% vs exact string match 3.4%.
- Made the comparison **structurally** fair rather than carefully fair: one
  evaluation script serves both runs with `--adapter` as the only difference,
  and training and evaluation share a single prompt builder with a test
  asserting the prompts are byte-identical.
- **Verified the reference data before trusting it.** Executing the
  ground-truth SQL revealed that case-sensitive matching returned a usable
  result for only 55.2% of examples — the reference itself was broken almost
  half the time. `COLLATE NOCASE` took it to 94.4%.
- **Self-tested the scoring harness** against inputs whose correct score was
  known in advance. This caught a SQLite misfeature where an unknown
  double-quoted identifier silently becomes a string literal instead of
  raising an error, which would have reported 100% validity for entirely wrong
  queries.
- **Trained at three data scales, all at the same epoch count**, rather than
  a single before/after point or an epoch-confounded pair, turning the result
  into a real diminishing-returns curve.
- **Followed up with a structured code review** of the finished pipeline
  across five axes rather than assuming a green test suite meant it was safe
  to keep running. That review caught two issues the tests had missed: a
  scaling study step that could silently reuse a stale evaluation result
  after retraining, and a comparison script that would crash instead of
  degrade on a run with zero executing queries. Fixed both and confirmed all
  139 tests still passed before merging.

**Result.** Execution accuracy **39.8% → 85.4%** (+45.6 points); syntactic
validity 55.2% → 99.8%; `sql_error` failures 219 → 0. Significant under a
paired McNemar test (238 fixed, 10 broken, χ² = 207.8, p < 1e-15), with 0
table overlap between train and eval. Accuracy *conditional on the query
executing* also rose (72.1% → 85.6%), so the gain was not merely queries
becoming runnable. The unexpected finding: 200 examples and 12 minutes of
training captured 82% of the total gain; 1,000 examples captured 92%.

---

## Anticipated interview questions

**"How do you know the improvement is real and not noise?"**
Both models were scored on the same 500 examples, so the outcomes are paired
and McNemar's test applies — it uses only the discordant pairs. 238 examples
were fixed and 10 broken, χ² = 207.8, p < 1e-15. The 95% Wilson intervals
don't overlap. And I report the 10 regressions, because a comparison showing
only improvements isn't a measurement.

**"Could the model have memorised the test set?"**
No tables are shared between the 6,000 training and 500 evaluation examples —
verified, not assumed, even though WikiSQL's official splits are disjoint by
construction. One question *string* appears in both, but against a different
table with different gold SQL. I used the official splits precisely so there
was no self-made split to defend.

**"Isn't 85% just the model learning to quote identifiers?"**
Partly, and that's most of it — `sql_error` went 219 → 0. But I separated the
two effects: accuracy among queries that actually executed also rose, 72.1% →
85.6%. So both the syntax and the column/aggregate selection improved. I'd
also flag that the exact-match jump to 77.8% partly reflects the model
learning my specific SQL rendering convention, which is why I report execution
accuracy as the primary metric.

**"Why not a bigger model, or full fine-tuning?"**
4 GB of VRAM. Full fine-tuning of even this 1.5B model needs roughly 24 GB.
I'd rather show a constraint handled honestly than claim a setup I didn't
have. The methodology transfers directly to a larger model on better hardware.

**"What would you do differently?"**
Two things remain open. I'd add a held-out dev-set evaluation during training
rather than only a 3-example smoke test. And WikiSQL is single-table, so the
obvious next step is Spider, where joins make the task genuinely harder. A
third thing I already went back and fixed: the data-efficiency curve
originally had only two points and an epoch confound, since n=200 ran 1
epoch while n=6,000 ran 2. I re-ran n=200 at 2 epochs and added an n=1,000
point — the share of total gain moved from 79% to 82% at n=200, with 92% at
n=1,000, confirming the direction but correcting the exact number.

**"What broke, and what did you do about it?"**
The most instructive one: my own scoring harness. Testing it against
known-answer inputs revealed SQLite silently converts an unknown double-quoted
identifier into a string literal, so a hallucinated column returned data
instead of an error — 100% validity at 0% accuracy. I added a schema check.
Then the baseline's real output showed the fix was too aggressive: the model
writes `WHERE No. = "21"`, a misquoted *value* that SQLite executes correctly.
Rejecting it would have penalised the baseline and inflated the fine-tuned
model's gain, so the check now distinguishes identifier position from value
position. That's the sort of error that doesn't announce itself — it just
produces a confident, wrong number.

A second one, later: I ran a structured code review against my own finished
pipeline across five axes (correctness, readability, architecture, security,
performance) rather than treating a passing test suite as proof it was safe
to keep running. It found two latent issues: the scaling study script would
silently reuse a stale evaluation result if training was rerun at a
different epoch count and the old output wasn't deleted, and the comparison
script divided by zero with no guard if a run ever produced zero executing
queries. Neither had produced a wrong number yet, but both were one rerun
away from it. I fixed both and confirmed all 139 tests still passed before
merging.

---

## Honest framing notes

Things to say before being asked, because volunteering them reads as
confidence rather than concession:

- WikiSQL is **single-table** — no joins. This is not general enterprise
  text-to-SQL, which is a substantially harder problem.
- The evaluation set **excludes ~8% degenerate examples** whose ground truth
  returns nothing, so the figure isn't directly comparable to published
  leaderboard numbers.
- It is a **methodology demonstration**, not a shippable product — no serving
  layer, no ambiguity handling.
- The **exact-match metric is inflated** by the model learning this project's
  SQL formatting convention. Execution accuracy is the defensible number.
