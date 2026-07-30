"""Model loading and batched generation.

Deliberately model-agnostic: the same functions load the quantised base model
for the Phase 2 baseline and the base model plus a LoRA adapter for Phase 4.
The only thing that differs between those two runs is the `adapter_path`
argument, which is what makes the before/after comparison structurally
apples-to-apples rather than a matter of discipline.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "model.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or CONFIG_PATH).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model_and_tokenizer(
    config: dict[str, Any],
    adapter_path: str | Path | None = None,
    model_id: str | None = None,
):
    """Load the 4-bit base model, optionally with a LoRA adapter attached."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization = config["quantization"]
    runtime = config["runtime"]
    resolved_id = model_id or config["model"]["id"]

    supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization["load_in_4bit"],
        bnb_4bit_quant_type=quantization["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quantization["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Pass the pinned revision through. Without this the config's `revision`
    # is decorative: `from_pretrained` defaults to `main`, and the base model
    # could change between the baseline and post-fine-tuning runs.
    revision = config["model"].get("revision") or "main"

    tokenizer = AutoTokenizer.from_pretrained(resolved_id, revision=revision)
    # Decoder-only models must be left-padded for batched generation: with
    # right padding the pad tokens sit between the prompt and the first
    # generated token, and the continuation is conditioned on padding.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Annotated Any: the returned model is a PeftModel when an adapter is
    # attached and the bare base model otherwise -- a real union of dynamic
    # transformers/peft types not worth modelling precisely here.
    model: Any = AutoModelForCausalLM.from_pretrained(
        resolved_id,
        revision=revision,
        quantization_config=bnb_config,
        device_map=runtime["device_map"],
        dtype="auto",
        attn_implementation=runtime["attn_implementation"],
    )

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def generate_batched(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    max_input_tokens: int = 512,
    progress: bool = True,
) -> Iterator[str]:
    """Greedy-decode each prompt, yielding only the newly generated text.

    Greedy (`do_sample=False`) on purpose: sampling would make the evaluation
    non-deterministic, and a before/after comparison that moves when rerun is
    not a measurement. Temperature is left unset for the same reason.
    """
    import torch

    total = len(prompts)
    for start in range(0, total, batch_size):
        batch = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_length = encoded["input_ids"].shape[1]
        for sequence in generated:
            yield tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)

        if progress:
            done = min(start + batch_size, total)
            print(f"    generated {done}/{total}", end="\r", flush=True)
    if progress:
        print()
