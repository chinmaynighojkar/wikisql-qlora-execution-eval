"""Prompt construction.

One template, used everywhere: the baseline run, the fine-tuning targets, and
the post-fine-tuning run. If the baseline saw a different prompt from the
fine-tuned model, the before/after comparison would measure prompt
engineering rather than fine-tuning, so the template lives here alone and is
never inlined at a call site.

Column names are presented **quoted** because 1,116 distinct headers in the
corpus contain characters that are illegal in a bare SQL identifier
(`School/Club Team`, `Pick #`). Showing them quoted is what makes correct
quoting the obvious continuation rather than something the model must infer.
"""

from __future__ import annotations

from typing import Any, Sequence

SYSTEM_PROMPT = (
    "You are an expert at writing SQLite queries. "
    "Given a table schema and a question, reply with exactly one SQLite "
    "SELECT statement that answers it. "
    "Output only the SQL. No explanation, no markdown, no code fences."
)

# WikiSQL types map onto exactly two SQLite storage classes.
_TYPE_LABELS = {"real": "REAL", "text": "TEXT"}


def format_schema(table_name: str, columns: Sequence[str], column_types: Sequence[str]) -> str:
    lines = [f'Table: "{table_name}"', "Columns:"]
    for column, type_name in zip(columns, column_types):
        lines.append(f'  "{column}" ({_TYPE_LABELS.get(type_name, "TEXT")})')
    return "\n".join(lines)


def build_user_message(record: dict[str, Any]) -> str:
    schema = format_schema(record["table_name"], record["columns"], record["column_types"])
    return f"{schema}\n\nQuestion: {record['question']}"


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Chat-format messages for an instruct model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(record)},
    ]


def build_prompt(tokenizer: Any, record: dict[str, Any]) -> str:
    """Render the prompt through the model's own chat template.

    Using the tokenizer's template rather than a hand-written string keeps the
    special tokens correct for the model actually being evaluated, and means
    the baseline and fine-tuned runs are formatted identically by
    construction.
    """
    return tokenizer.apply_chat_template(
        build_messages(record),
        tokenize=False,
        add_generation_prompt=True,
    )


def build_training_text(tokenizer: Any, record: dict[str, Any]) -> str:
    """Prompt plus the gold SQL as the assistant turn, for Phase 3 SFT."""
    messages = build_messages(record) + [
        {"role": "assistant", "content": record["gold_sql"]}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)
