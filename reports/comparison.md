# Before / after: QLoRA fine-tuning for text-to-SQL

Evaluated on **500 held-out WikiSQL test examples**, scored by execution match against real SQLite tables. Identical eval code for both runs -- the only difference is the LoRA adapter.

## Headline

| Metric | Base model | + QLoRA adapter | Change |
|---|---:|---:|---:|
| **Execution accuracy** (primary) | 39.8% | 85.4% | **+45.6 pts** |
| Syntactic validity | 55.2% | 99.8% | **+44.6 pts** |
| Exact string match (secondary) | 3.4% | 77.8% | **+74.4 pts** |
| SQL extraction rate | 100.0% | 100.0% | **+0.0 pts** |

95% Wilson intervals on execution accuracy: baseline [35.6%, 44.1%], fine-tuned [82.0%, 88.2%] — non-overlapping.

## Training scale

The adapter was trained on **6,000 examples** for **2 epoch(s)** in **6711 seconds**, peaking at **2.628 GiB** of VRAM on a 4 GB RTX 3050 Laptop. 18,464,768 trainable parameters (2.0356% of the model).

## Is the difference real?

Both models were scored on the same examples, so the outcomes are paired and McNemar's test applies:

| | Fine-tuned correct | Fine-tuned wrong |
|---|---:|---:|
| **Baseline correct** | 189 | 10 |
| **Baseline wrong** | 238 | 63 |

Fine-tuning fixed **238** examples and broke **10**. chi-square = 207.778 (continuity corrected), p = <1e-15.

## Where the gain came from

If every gained point came from queries merely becoming runnable, the model would have learned SQL syntax and nothing about the question. Accuracy among queries that actually executed:

| | Executed | Correct | Accuracy given execution |
|---|---:|---:|---:|
| Base model | 276 | 199 | 72.1% |
| + adapter | 499 | 427 | 85.6% |

So the improvement is **both** effects: far more queries run (validity 55.2% -> 99.8%), and those that run are also more often right (72.1% -> 85.6%).

### Failure modes

| Failure | Base model | + adapter |
|---|---:|---:|
| `sql_error` | 219 | 0 |
| `unknown_identifier` | 5 | 1 |
| `wrong_result` | 77 | 72 |

## Leakage checks

- Tables shared between the training subsample and the evaluation set: **0** (of 5,179 train and 483 eval tables). WikiSQL's official splits have disjoint tables; verified rather than assumed.
- Question strings appearing in both: **1**, of which **0** have identical gold SQL.

## Qualitative examples

### Fixed by fine-tuning

**Q:** What is the lowest attendance when footscray is the away team?

```sql
-- gold
SELECT MIN("Crowd") FROM "table_2_10640687_18" WHERE "Away team" = 'footscray'
-- base model  (sql_error: OperationalError: no such column: Away_team)
SELECT MIN(Crowd) AS min_attendance FROM table_2_10640687_18 WHERE Away_team = 'footscray'
-- + adapter   (correct)
SELECT MIN("Crowd") FROM "table_2_10640687_18" WHERE "Away team" = 'footscray'
```

**Q:** Which charts had debut sales of of more than 339333.011497678?

```sql
-- gold
SELECT "Oricon Albums Chart" FROM "table_23180638_1" WHERE "Debut Sales (copies)" > 339333.011497678
-- base model  (sql_error: OperationalError: near "Chart": syntax error)
SELECT Oricon Albums Chart FROM table_23180638_1 WHERE Debut Sales (copies) > 339333.011497678
-- + adapter   (correct)
SELECT "Oricon Albums Chart" FROM "table_23180638_1" WHERE "Debut Sales (copies)" > 339333.011497678
```

**Q:** What is the original airdate when the season number is 4?

```sql
-- gold
SELECT "Original air date" FROM "table_15824796_5" WHERE "Season #" = 4
-- base model  (sql_error: OperationalError: unrecognized token: "#")
SELECT Original_air_date FROM table_15824796_5 WHERE Season_# = 4
-- + adapter   (correct)
SELECT "Original air date" FROM "table_15824796_5" WHERE "Season #" = 4
```

### Broken by fine-tuning

Reported for the same reason the fixes are: a comparison that only showed improvements would not be a measurement.

**Q:** Who was Kraco Twin 125 (R2)'s Winning Driver?

```sql
-- gold
SELECT "Winning driver" FROM "table_2_10706879_3" WHERE "Name" = 'kraco twin 125 (r2)'
-- base model  (correct)
SELECT `Winning driver` FROM `table_2_10706879_3` WHERE `Name` = 'Kraco Twin 125 (R2)'
-- + adapter   (wrong_result)
SELECT "Winning driver" FROM "table_2_10706879_3" WHERE "Fastest Lap" = 'kraco twin 125 (r2)'
```

**Q:** I want the lowest Gold for silver being 2 and electronic sports

```sql
-- gold
SELECT MIN("Gold") FROM "table_2_10882501_8" WHERE "Silver" = 2 AND "Sport" = 'electronic sports'
-- base model  (correct)
SELECT MIN("Gold") AS "Lowest_Gold" FROM "table_2_10882501_8" WHERE "Silver" = 2 AND "Sport" LIKE "%electronic%"
-- + adapter   (wrong_result)
SELECT MIN("Gold") FROM "table_2_10882501_8" WHERE "Silver" = 2 AND "Sport" = 'electronic'
```

**Q:** How many points total are there later than 2003?

```sql
-- gold
SELECT SUM("Points") FROM "table_2_1618788_2" WHERE "Year" > 2003
-- base model  (correct)
SELECT SUM(Points) AS TotalPoints FROM table_2_1618788_2 WHERE Year > 2003
-- + adapter   (wrong_result)
SELECT COUNT("Points") FROM "table_2_1618788_2" WHERE "Year" > 2003
```
