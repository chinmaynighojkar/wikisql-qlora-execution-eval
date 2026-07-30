"""Deliberate seeding for every entry point that touches randomness.

Results here are already reproducible in practice -- `SFTConfig(seed=...)`
covers the trainer, and greedy decoding makes evaluation deterministic on its
own -- but that reproducibility was a property of the current configuration,
not a decision anyone made. It would silently stop being true the day someone
switches evaluation to sampled decoding. Calling `seed_everything` at each
entry point makes it a decision that survives that change.
"""

from __future__ import annotations

import random


def seed_everything(seed: int) -> None:
    """Seed every source of randomness this project's entry points can reach."""
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
