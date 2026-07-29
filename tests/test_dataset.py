"""Tests for SFT dataset construction.

The prompt-parity tests here guard the quietest failure mode in the whole
project. If the fine-tuned model is trained on a prompt format that differs
from the one the evaluation harness uses -- even by a stray newline -- then
Phase 4 measures a mixture of learning and format mismatch, and nothing in the
output would look wrong. The number would simply be misleading.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_text_to_sql.dataset import to_sft_example, token_length_stats  # noqa: E402
from lora_text_to_sql.prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_messages,
    build_prompt,
    build_user_message,
    format_schema,
)

RECORD = {
    "table_id": "1-1-1",
    "table_name": "table_1_1_1",
    "columns": ["Player", "School/Club Team", "Points"],
    "column_types": ["text", "text", "real"],
    "question": "Who scored the most?",
    "gold_sql": 'SELECT "Player" FROM "table_1_1_1" WHERE "Points" > 10',
    "agg_index": 0,
    "condition_count": 1,
}


class StubTokenizer:
    """Minimal stand-in for a chat tokenizer.

    Mimics ChatML (Qwen2.5's format) closely enough to test prompt assembly
    without pulling in torch or downloading a real tokenizer.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        rendered = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


@pytest.fixture
def tokenizer():
    return StubTokenizer()


# --------------------------------------------------------------------------
# Prompt parity
# --------------------------------------------------------------------------


class TestPromptParity:
    def test_training_prompt_matches_eval_prompt_exactly(self, tokenizer):
        """The eval prompt must be a byte-exact prefix of the training text.

        Training renders prompt + completion; evaluation renders prompt +
        generation marker. If the first diverges from the second, the model is
        being asked at eval time for a continuation of something it never saw
        in training.
        """
        eval_prompt = build_prompt(tokenizer, RECORD)

        example = to_sft_example(RECORD)
        training_text = tokenizer.apply_chat_template(
            example["prompt"], add_generation_prompt=True
        )

        assert training_text == eval_prompt

    def test_both_paths_use_the_same_message_builder(self):
        """Parity is structural: one function, two callers."""
        assert to_sft_example(RECORD)["prompt"] == build_messages(RECORD)

    def test_full_training_text_extends_the_eval_prompt(self, tokenizer):
        example = to_sft_example(RECORD)
        full = tokenizer.apply_chat_template(example["prompt"] + example["completion"])
        eval_prompt = build_prompt(tokenizer, RECORD)
        assert full.startswith(eval_prompt.removesuffix("<|im_start|>assistant\n"))
        assert RECORD["gold_sql"] in full


# --------------------------------------------------------------------------
# Example shape
# --------------------------------------------------------------------------


class TestToSftExample:
    def test_has_trl_prompt_completion_keys(self):
        """TRL detects prompt-completion datasets by the `prompt` key, and
        applies completion-only loss to them."""
        example = to_sft_example(RECORD)
        assert set(example) == {"prompt", "completion"}

    def test_completion_is_the_assistant_turn_only(self):
        completion = to_sft_example(RECORD)["completion"]
        assert completion == [{"role": "assistant", "content": RECORD["gold_sql"]}]

    def test_prompt_never_contains_the_answer(self):
        """If the gold SQL leaked into the prompt, training loss would collapse
        and the model would learn to copy rather than translate."""
        prompt_text = " ".join(m["content"] for m in to_sft_example(RECORD)["prompt"])
        assert RECORD["gold_sql"] not in prompt_text
        assert "SELECT" not in prompt_text.replace(SYSTEM_PROMPT, "")

    def test_roles_are_ordered_system_then_user(self):
        assert [m["role"] for m in build_messages(RECORD)] == ["system", "user"]


# --------------------------------------------------------------------------
# Schema rendering
# --------------------------------------------------------------------------


class TestSchemaRendering:
    def test_columns_are_quoted_in_the_prompt(self):
        """1,116 headers in the corpus are illegal as bare identifiers, so the
        prompt shows them quoted -- making correct quoting the obvious
        continuation rather than something the model must infer."""
        schema = format_schema("t", ["School/Club Team"], ["text"])
        assert '"School/Club Team"' in schema

    def test_types_are_rendered_as_sqlite_storage_classes(self):
        schema = format_schema("t", ["A", "B"], ["real", "text"])
        assert "(REAL)" in schema and "(TEXT)" in schema

    def test_question_is_present(self):
        assert RECORD["question"] in build_user_message(RECORD)

    def test_unknown_type_falls_back_to_text(self):
        assert "(TEXT)" in format_schema("t", ["A"], ["mystery"])


# --------------------------------------------------------------------------
# Length statistics
# --------------------------------------------------------------------------


def test_token_length_stats_shape(tokenizer):
    stats = token_length_stats(tokenizer, [RECORD] * 10)
    assert stats["n"] == 10
    assert stats["min"] <= stats["median"] <= stats["max"]
