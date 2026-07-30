"""Build the supervised fine-tuning dataset.

The single most important property here is **prompt parity**: the messages a
training example is built from are produced by exactly the same
`prompt.build_messages()` the evaluation harness uses. If training saw a
different prompt format from evaluation, Phase 4 would partly measure format
mismatch rather than learning, and the direction of that error is unknowable
without a third experiment.

Parity is structural rather than careful -- there is one function, and both
paths call it. `tests/test_dataset.py` asserts byte-identical prompts.

Format: TRL's *conversational prompt-completion* type.

    {"prompt":     [ {system...}, {user...} ],
     "completion": [ {"role": "assistant", "content": "SELECT ..."} ]}

TRL detects this by the presence of a `prompt` key and, for prompt-completion
datasets, computes loss on the completion only (`completion_only_loss`
defaults to True for this dataset type). That is what we want: the model
should be scored on producing the SQL, not on reciting the schema it was
handed. Training on prompt tokens spends capacity memorising table
definitions that are given at inference time anyway.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .prompt import build_messages


def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Thin alias kept for call-site readability; see `io.read_jsonl`."""
    return read_jsonl(path, limit)


def to_sft_example(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Convert one Phase 1 record into a TRL prompt-completion example."""
    return {
        "prompt": build_messages(record),
        "completion": [{"role": "assistant", "content": record["gold_sql"]}],
    }


def build_sft_dataset(records: Iterable[dict[str, Any]]):
    """Materialise a `datasets.Dataset` of prompt-completion examples."""
    from datasets import Dataset

    return Dataset.from_list([to_sft_example(record) for record in records])


def token_length_stats(tokenizer: Any, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Measure prompt+completion token lengths against the configured max.

    Run before training rather than after. Silent truncation at `max_length`
    would cut the tail off the gold SQL, so the model would be trained toward
    incomplete queries -- which shows up as a mediocre Phase 4 number with no
    obvious cause. Better to see the distribution first.
    """
    lengths = []
    for record in records:
        text = tokenizer.apply_chat_template(
            build_messages(record)
            + [{"role": "assistant", "content": record["gold_sql"]}],
            tokenize=False,
        )
        lengths.append(len(tokenizer(text)["input_ids"]))

    lengths.sort()
    count = len(lengths)
    return {
        "n": count,
        "min": lengths[0],
        "median": lengths[count // 2],
        "p95": lengths[int(count * 0.95)],
        "max": lengths[-1],
        "mean": round(sum(lengths) / count, 1),
    }
