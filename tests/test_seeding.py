"""Tests for deliberate seeding.

Results were already reproducible by accident (SFTConfig(seed=...) plus
greedy decoding); these tests confirm seed_everything actually controls
randomness rather than assuming it does.
"""

import random

import pytest

from lora_text_to_sql.seeding import seed_everything


class TestStdlibRandom:
    def test_same_seed_reproduces_the_sequence(self):
        seed_everything(20260728)
        first = [random.random() for _ in range(5)]
        seed_everything(20260728)
        second = [random.random() for _ in range(5)]
        assert first == second

    def test_different_seeds_diverge(self):
        seed_everything(1)
        a = [random.random() for _ in range(5)]
        seed_everything(2)
        b = [random.random() for _ in range(5)]
        assert a != b


class TestTorchSeeding:
    def test_same_seed_reproduces_a_random_tensor(self):
        torch = pytest.importorskip("torch")
        seed_everything(20260728)
        first = torch.rand(4)
        seed_everything(20260728)
        second = torch.rand(4)
        assert torch.equal(first, second)

    def test_missing_torch_does_not_raise(self, monkeypatch):
        """seed_everything must degrade gracefully in a torch-free
        environment (e.g. this project's own CI, which deliberately does not
        install torch) rather than crash the whole entry point on import."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated: torch not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        seed_everything(20260728)  # must not raise
