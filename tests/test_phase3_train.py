"""Tests for the extracted, testable stages of phase3_train.main().

Before this decomposition, phase3_train.main() was one ~215-line function
doing config resolution, data loading, model loading, training, and report
assembly in sequence -- nothing in it except assess_loss_trend (already
tested in test_loss_trend.py) could be tested without a GPU and a downloaded
model. These stages are pure, so they can be tested without either.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_phase3():
    spec = importlib.util.spec_from_file_location(
        "phase3_train", REPO_ROOT / "scripts" / "phase3_train.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase3_train"] = module
    spec.loader.exec_module(module)
    return module


phase3 = load_phase3()


def _args(**overrides):
    defaults = {"rank": None, "attention_only": False, "epochs": None, "logging_steps": None}
    defaults.update(overrides)
    return phase3.argparse.Namespace(**defaults)


class TestArgParser:
    def test_defaults(self):
        args = phase3.build_arg_parser().parse_args([])
        assert args.limit is None
        assert args.rank is None
        assert args.attention_only is False
        assert args.output == str(phase3.ADAPTER_DIR)


class TestApplyOverrides:
    LORA: ClassVar[dict[str, Any]] = {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
    SETTINGS: ClassVar[dict[str, Any]] = {"num_train_epochs": 2}

    def test_no_overrides_leaves_settings_unchanged(self):
        lora, settings = phase3.apply_overrides(_args(), self.LORA, self.SETTINGS)
        assert lora == self.LORA
        assert settings == self.SETTINGS

    def test_does_not_mutate_the_originals(self):
        """The YAML-loaded dicts must not be mutated in place -- a second
        call (e.g. a scaling study looping over sizes) must start clean."""
        original_lora = dict(self.LORA)
        phase3.apply_overrides(_args(rank=8), self.LORA, self.SETTINGS)
        assert self.LORA == original_lora

    def test_rank_override_also_sets_alpha_to_2x(self):
        lora, _ = phase3.apply_overrides(_args(rank=8), self.LORA, self.SETTINGS)
        assert lora["r"] == 8
        assert lora["lora_alpha"] == 16

    def test_attention_only_drops_mlp_projections(self):
        lora, _ = phase3.apply_overrides(_args(attention_only=True), self.LORA, self.SETTINGS)
        assert lora["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_epochs_override(self):
        _, settings = phase3.apply_overrides(_args(epochs=1.0), self.LORA, self.SETTINGS)
        assert settings["num_train_epochs"] == 1.0


class TestResolveLoggingSteps:
    SETTINGS: ClassVar[dict[str, Any]] = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "logging_steps": 10,
    }

    def test_short_run_lowers_logging_steps(self):
        """The exact regression this exists for: a 200-record, 1-epoch smoke
        run produces ~12 optimiser steps, which at logging_steps=10 logs a
        single point and makes the Phase 3 gate vacuous."""
        settings = {**self.SETTINGS, "num_train_epochs": 1.0}
        logging_steps, optimizer_steps = phase3.resolve_logging_steps(_args(), settings, 200)
        assert optimizer_steps == 12
        assert logging_steps < 10
        assert logging_steps >= 1

    def test_long_run_keeps_configured_logging_steps(self):
        settings = {**self.SETTINGS, "num_train_epochs": 2}
        logging_steps, optimizer_steps = phase3.resolve_logging_steps(_args(), settings, 6000)
        assert optimizer_steps == 750
        assert logging_steps == 10

    def test_explicit_override_always_wins(self):
        settings = {**self.SETTINGS, "num_train_epochs": 1.0}
        logging_steps, _ = phase3.resolve_logging_steps(_args(logging_steps=3), settings, 200)
        assert logging_steps == 3

    def test_optimizer_steps_floor_is_one(self):
        settings = {**self.SETTINGS, "num_train_epochs": 0.001}
        _, optimizer_steps = phase3.resolve_logging_steps(_args(), settings, 1)
        assert optimizer_steps >= 1


class TestComputeVerdict:
    def _smoke(self, executed_flags, match_flags=None):
        match_flags = match_flags or executed_flags
        return [
            {"executed": e, "execution_match": m}
            for e, m in zip(executed_flags, match_flags)
        ]

    def test_pass_requires_decreasing_loss_and_all_executable(self):
        losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        verdict = phase3.compute_verdict(losses, self._smoke([True, True, True]))
        assert verdict["verdict"] == "PASS"
        assert verdict["passed"] is True

    def test_fail_when_smoke_test_has_invalid_sql(self):
        losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        verdict = phase3.compute_verdict(losses, self._smoke([True, False, True]))
        assert verdict["verdict"] == "FAIL"
        assert verdict["passed"] is False
        assert verdict["valid_sql"] == 2

    def test_fail_when_loss_does_not_decrease(self):
        losses = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        verdict = phase3.compute_verdict(losses, self._smoke([True, True, True]))
        assert verdict["verdict"] == "FAIL"

    def test_indeterminate_on_too_few_loss_points(self):
        """Too short a run to judge a trend must not silently count as a
        pass or a fail -- this is the exact bug test_loss_trend.py guards."""
        verdict = phase3.compute_verdict([0.13], self._smoke([True, True, True]))
        assert verdict["verdict"] == "INDETERMINATE"
        assert verdict["passed"] is False


class TestBuildSftConfig:
    def test_bf16_and_fp16_are_complementary(self):
        pytest.importorskip("trl")
        settings = {
            "num_train_epochs": 2, "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16, "gradient_checkpointing": True,
            "optim": "paged_adamw_8bit", "learning_rate": 0.0002,
            "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
            "max_grad_norm": 0.3, "weight_decay": 0.0, "max_length": 512,
            "logging_steps": 10, "save_strategy": "no", "seed": 1, "packing": False,
        }
        config = phase3.build_sft_config(settings, Path("models/_checkpoints"), bf16=True)
        assert config.bf16 is True
        assert config.fp16 is False

        config = phase3.build_sft_config(settings, Path("models/_checkpoints"), bf16=False)
        assert config.bf16 is False
        assert config.fp16 is True
