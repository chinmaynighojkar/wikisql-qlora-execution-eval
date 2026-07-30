"""Phase 0 gate: prove Qwen2.5-1.5B-Instruct loads in 4-bit NF4 and runs a
forward pass inside 4 GB of VRAM on an RTX 3050 Laptop.

This script is a *gate*, not a demo. It either passes with real measured
numbers written to reports/phase0_env_report.json, or it fails loudly with the
reason. Nothing downstream (dataset prep, baseline eval, training) should be
built until it passes, because every later phase assumes the 4-bit path works.

Two failure modes it is specifically designed to catch, both of which would
otherwise produce a *false pass*:

  1. Silent CPU offload. `device_map="auto"` moves layers to CPU RAM when VRAM
     runs short, so the script "succeeds" while proving nothing about the 4 GB
     budget. We pin `device_map={"": 0}` and then assert every parameter is
     actually on the GPU.

  2. Windows CUDA sysmem fallback. Recent NVIDIA Windows drivers can spill
     VRAM into system RAM rather than raising OOM, which looks like a pass but
     will be catastrophically slow during training. We report device-wide VRAM
     from `mem_get_info()` alongside torch's own allocator counters; a large
     gap between them, or a device-wide figure near the card's limit, is the
     signal. See docs/SETUP.md.

Usage:
    python scripts/phase0_verify_env.py
    python scripts/phase0_verify_env.py --train-probe   # optional, see below
    python scripts/phase0_verify_env.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "model.yaml"
REPORT_PATH = REPO_ROOT / "reports" / "phase0_env_report.json"

BYTES_PER_GIB = 1024**3

# A representative Phase 2 prompt. Using a realistic text-to-SQL prompt rather
# than "Hello world" matters: the forward-pass memory figure is only meaningful
# at the sequence length the project will actually use.
SAMPLE_PROMPT = """You are a SQL generator. Given a table schema and a question, respond with a single SQLite SELECT statement and nothing else.

Table: table_1_10015132_16
Columns:
  Player (text)
  No. (text)
  Nationality (text)
  Position (text)
  Years_in_Toronto (text)
  School_Club_Team (text)

Question: What school did player number 21 come from?

SQL:"""


# --------------------------------------------------------------------------
# Result plumbing
# --------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    # `name`, `passed` and `detail` are positional-only (the `/`) so that a
    # payload key of the same name lands in **data instead of colliding with
    # the parameter. Without this, `report.add("x", True, "y", **{"name": ...})`
    # raises "got multiple values for argument 'name'".
    def add(self, name: str, passed: bool, detail: str, /, **data: Any) -> Check:
        check = Check(name=name, passed=passed, detail=detail, data=data)
        self.checks.append(check)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        for key, value in data.items():
            print(f"           {key} = {value}")
        return check

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": 0,
            "purpose": "Verify 4-bit NF4 load + forward pass fits in available VRAM",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "overall_result": "PASS" if self.ok else "FAIL",
            "checks": [asdict(c) for c in self.checks],
        }


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def gib(num_bytes: float) -> float:
    return round(num_bytes / BYTES_PER_GIB, 3)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    import yaml

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_interpreter(report: Report) -> None:
    section("1. Interpreter and platform")
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    is_310 = sys.version_info[:2] == (3, 10)
    report.add(
        "python_version",
        # 3.10 is the project's target (matches the working setup on the sibling
        # projects). Anything >= 3.10 will import, so this is a warning-level
        # check rather than a hard gate, but the mismatch is recorded.
        sys.version_info >= (3, 10),
        f"Python {version}" + ("" if is_310 else "  (project targets 3.10)"),
        version=version,
        matches_target_3_10=is_310,
        executable=sys.executable,
    )
    report.add(
        "platform",
        True,
        f"{platform.system()} {platform.release()}",
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
    )


def check_torch_and_gpu(report: Report) -> tuple[Any, dict[str, Any]]:
    section("2. PyTorch and CUDA device")
    try:
        import torch
    except ImportError as exc:
        report.add("torch_import", False, f"torch not installed: {exc}")
        return None, {}

    report.add(
        "torch_import",
        True,
        f"torch {torch.__version__}",
        version=torch.__version__,
        cuda_build=torch.version.cuda,
    )

    if torch.version.cuda is None:
        report.add(
            "torch_cuda_build",
            False,
            "CPU-only torch build installed. Reinstall from the CUDA index "
            "(see docs/SETUP.md) -- a CPU wheel cannot run 4-bit bitsandbytes.",
        )
        return torch, {}
    report.add("torch_cuda_build", True, f"CUDA build {torch.version.cuda}")

    if not torch.cuda.is_available():
        report.add(
            "cuda_available",
            False,
            "torch.cuda.is_available() is False -- no usable GPU visible. "
            "Check the NVIDIA driver and that no other process holds the device.",
        )
        return torch, {}

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    capability = f"{props.major}.{props.minor}"
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)

    gpu_info = {
        "device_name": props.name,
        "compute_capability": capability,
        "total_vram_gib": gib(total_bytes),
        "free_vram_gib_at_start": gib(free_bytes),
    }
    report.add(
        "cuda_available",
        True,
        f"{props.name} ({gpu_info['total_vram_gib']} GiB total, "
        f"{gpu_info['free_vram_gib_at_start']} GiB free)",
        **gpu_info,
    )

    # bf16 needs compute capability >= 8.0. Ampere (RTX 3050 = sm_86) qualifies.
    # Detected rather than assumed so the script stays correct on other cards.
    supports_bf16 = props.major >= 8
    report.add(
        "bf16_support",
        True,  # informational: fp16 fallback exists, so this never blocks
        "bf16 supported" if supports_bf16 else "bf16 unsupported, will use fp16",
        supports_bf16=supports_bf16,
    )
    gpu_info["supports_bf16"] = supports_bf16
    return torch, gpu_info


def check_bitsandbytes(report: Report) -> bool:
    """The single biggest platform risk in this project.

    bitsandbytes has historically been unreliable on native Windows. Importing
    it is not enough -- the import can succeed while the compiled CUDA backend
    is missing. We force a real 4-bit quantise/dequantise round-trip on the GPU.
    """
    section("3. bitsandbytes 4-bit backend")
    try:
        import bitsandbytes as bnb
    except Exception as exc:  # noqa: BLE001 - bnb raises many exception types
        report.add(
            "bitsandbytes_import",
            False,
            f"import failed: {type(exc).__name__}: {exc}  -> see the WSL2 "
            "fallback in docs/SETUP.md",
        )
        return False

    report.add(
        "bitsandbytes_import",
        True,
        f"bitsandbytes {bnb.__version__}",
        version=bnb.__version__,
    )

    try:
        import torch
        from bitsandbytes import functional as bnb_functional

        probe = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        quantized, state = bnb_functional.quantize_4bit(probe, quant_type="nf4")
        restored = bnb_functional.dequantize_4bit(quantized, state)
        error = (restored.float() - probe.float()).abs().mean().item()
        del probe, quantized, restored, state
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        report.add(
            "bitsandbytes_nf4_kernel",
            False,
            f"NF4 kernel failed on GPU: {type(exc).__name__}: {exc}  -> this is "
            "the documented trigger for the WSL2 fallback (docs/SETUP.md)",
        )
        return False

    report.add(
        "bitsandbytes_nf4_kernel",
        True,
        "NF4 quantise/dequantise round-trip succeeded on GPU",
        mean_abs_reconstruction_error=round(error, 5),
    )
    return True


def load_model_4bit(report: Report, config: dict[str, Any], model_id: str, supports_bf16: bool):
    section("4. Load model in 4-bit NF4")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_cfg = config["quantization"]
    runtime_cfg = config["runtime"]

    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    torch.cuda.reset_peak_memory_stats()
    free_before, _ = torch.cuda.mem_get_info()

    print(f"  downloading / loading {model_id} (first run pulls ~3 GB from the Hub)...")
    started = time.perf_counter()
    try:
        revision = config["model"].get("revision") or "main"
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=bnb_config,
            # Pinned to GPU 0 on purpose: "auto" would hide an OOM as a silent
            # CPU offload and invalidate this whole check.
            device_map=runtime_cfg["device_map"],
            # transformers >= 5 renamed `torch_dtype` to `dtype`; "auto" reads
            # the dtype recorded in the checkpoint config.
            dtype="auto",
            attn_implementation=runtime_cfg["attn_implementation"],
        )
    except torch.cuda.OutOfMemoryError as exc:
        report.add(
            "model_load_4bit",
            False,
            f"OOM while loading {model_id}: {exc}  -> fall back to "
            f"{config['model']['fallback_id']} (documented in docs/DECISIONS.md)",
            model_id=model_id,
        )
        return None, None
    except Exception as exc:  # noqa: BLE001
        report.add(
            "model_load_4bit",
            False,
            f"load failed: {type(exc).__name__}: {exc}",
            model_id=model_id,
        )
        return None, None

    load_seconds = round(time.perf_counter() - started, 1)
    free_after, _total = torch.cuda.mem_get_info()

    report.add(
        "model_load_4bit",
        True,
        f"loaded in {load_seconds}s",
        model_id=model_id,
        compute_dtype=str(compute_dtype),
        quant_type=quant_cfg["bnb_4bit_quant_type"],
        double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        load_seconds=load_seconds,
        weights_vram_gib_device_wide=gib(free_before - free_after),
        torch_allocated_gib=gib(torch.cuda.memory_allocated()),
    )

    # --- offload guard -----------------------------------------------------
    # If any parameter landed on CPU or stayed on meta, the fit is not real.
    stray = {}
    for name, param in model.named_parameters():
        device_type = param.device.type
        if device_type != "cuda":
            stray[name] = str(param.device)
    report.add(
        "no_cpu_offload",
        not stray,
        (
            "all parameters resident on GPU"
            if not stray
            else f"{len(stray)} parameter(s) offloaded -- the 4 GB fit is NOT proven"
        ),
        offloaded_sample=dict(list(stray.items())[:5]),
    )

    # `numel()` on a 4-bit weight reports the *packed* element count, because
    # bitsandbytes stores two 4-bit values per uint8. Summing it naively gives
    # ~0.889B for a 1.5B model, which reads like the wrong checkpoint loaded.
    # Both figures are reported: the logical count is the one that should be
    # compared against the model card.
    packed_params = 0
    logical_params = 0
    for param in model.parameters():
        count = param.numel()
        packed_params += count
        if param.__class__.__name__ == "Params4bit":
            # Two 4-bit values per byte of quant storage.
            storage = getattr(param, "quant_storage", None)
            itemsize = storage.itemsize if storage is not None else 1
            count = count * 2 * itemsize
        logical_params += count

    report.add(
        "parameter_count",
        True,
        f"{logical_params / 1e9:.3f}B logical parameters "
        f"({packed_params / 1e9:.3f}B stored elements after 4-bit packing)",
        logical_parameters=logical_params,
        packed_storage_elements=packed_params,
        note=(
            "numel() on a Params4bit tensor counts packed uint8 storage, not "
            "logical weights; two 4-bit values share each byte."
        ),
    )
    return model, tokenizer


def run_forward_pass(report: Report, model, tokenizer) -> None:
    section("5. Forward pass")
    import torch

    inputs = tokenizer(SAMPLE_PROMPT, return_tensors="pt").to(model.device)
    seq_len = int(inputs["input_ids"].shape[1])

    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()

    try:
        started = time.perf_counter()
        with torch.no_grad():
            # labels=input_ids gives a real loss value, which is a cheap sanity
            # check that the quantised weights are not producing garbage.
            outputs = model(**inputs, labels=inputs["input_ids"])
        torch.cuda.synchronize()
        elapsed = round(time.perf_counter() - started, 3)
    except torch.cuda.OutOfMemoryError as exc:
        report.add("forward_pass", False, f"OOM during forward pass: {exc}")
        return

    free_after, _ = torch.cuda.mem_get_info()
    loss = float(outputs.loss.item())

    # A quantised 1.5B instruct model on natural text should land well under
    # ~10 nats. A wildly high loss means the quantised weights are broken even
    # though nothing raised.
    loss_sane = 0.0 < loss < 20.0

    report.add(
        "forward_pass",
        True,
        f"completed in {elapsed}s on {seq_len} tokens",
        sequence_length=seq_len,
        logits_shape=list(outputs.logits.shape),
        seconds=elapsed,
    )
    report.add(
        "forward_pass_loss_sane",
        loss_sane,
        (
            f"cross-entropy loss {loss:.4f} is in the plausible range"
            if loss_sane
            else f"cross-entropy loss {loss:.4f} is implausible -- suspect broken quantisation"
        ),
        loss=round(loss, 4),
    )

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    device_wide_used = total - free_after

    report.add(
        "vram_headroom",
        free_after > 0,
        (
            f"{gib(device_wide_used)} GiB of {gib(total)} GiB in use after the "
            f"forward pass ({gib(free_after)} GiB free)"
        ),
        # torch's own allocator view (tensors only)
        peak_torch_allocated_gib=gib(peak_allocated),
        peak_torch_reserved_gib=gib(peak_reserved),
        # the honest number: includes the CUDA context and fragmentation
        device_wide_used_gib=gib(device_wide_used),
        device_wide_free_gib=gib(free_after),
        total_vram_gib=gib(total),
        forward_pass_delta_gib=gib(free_before - free_after),
    )

    del outputs
    torch.cuda.empty_cache()


def run_generation_smoke_test(report: Report, model, tokenizer) -> None:
    """Not part of the Phase 0 gate, but Phase 2 depends on generation working,
    so failing here now is cheaper than failing there later."""
    section("6. Generation smoke test (Phase 2 dependency)")
    import torch

    inputs = tokenizer(SAMPLE_PROMPT, return_tensors="pt").to(model.device)
    try:
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,  # greedy: deterministic, matches the eval harness
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = round(time.perf_counter() - started, 2)
    except Exception as exc:  # noqa: BLE001
        report.add("generation_smoke_test", False, f"{type(exc).__name__}: {exc}")
        return

    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    tokens_per_second = round(len(new_tokens) / elapsed, 1) if elapsed else 0.0

    print(f"\n  --- raw model output (untuned base model) ---\n  {text}\n")

    report.add(
        "generation_smoke_test",
        bool(text),
        f"generated {len(new_tokens)} tokens in {elapsed}s ({tokens_per_second} tok/s)",
        seconds=elapsed,
        tokens_per_second=tokens_per_second,
        # Recorded verbatim. This is the untuned model's unscored output, kept
        # only as evidence the pipeline runs -- it is NOT a baseline metric.
        # The real baseline is produced by the Phase 2 harness.
        sample_output=text,
    )


def run_train_probe(report: Report, model, tokenizer) -> None:
    """Optional: one LoRA forward+backward to estimate training-time VRAM.

    Deliberately excluded from the Phase 0 gate. An inference forward pass
    fitting in VRAM does NOT prove training fits -- gradients, activations and
    optimiser states add substantially. This probe gives an early read on
    Phase 3 feasibility, but the authoritative check is Phase 3 itself.
    """
    section("7. Optional training-memory probe (not a Phase 0 gate)")
    import torch

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        report.add("train_probe", False, f"peft not installed: {exc}")
        return

    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.gradient_checkpointing_enable()
        peft_model.enable_input_require_grads()

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())

        inputs = tokenizer(SAMPLE_PROMPT, return_tensors="pt").to(peft_model.device)
        torch.cuda.reset_peak_memory_stats()

        outputs = peft_model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()
        torch.cuda.synchronize()

        _, total_vram = torch.cuda.mem_get_info()
        peak = torch.cuda.max_memory_allocated()

        report.add(
            "train_probe",
            True,
            (
                f"one LoRA fwd+bwd (batch 1, {inputs['input_ids'].shape[1]} tokens) "
                f"peaked at {gib(peak)} GiB allocated"
            ),
            trainable_parameters=trainable,
            total_parameters=total,
            trainable_percent=round(100 * trainable / total, 4),
            peak_torch_allocated_gib=gib(peak),
            total_vram_gib=gib(total_vram),
            note=(
                "Batch 1 with gradient checkpointing and no optimiser states. "
                "Real training adds AdamW state for the LoRA params only "
                "(~8 bytes/trainable param). Indicative, not authoritative."
            ),
        )
        peft_model.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError as exc:
        report.add(
            "train_probe",
            False,
            f"OOM during LoRA fwd+bwd: {exc}  -> Phase 3 will need a smaller "
            "rank, shorter sequences, or the 0.5B fallback model",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("train_probe", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="override the model id from configs/model.yaml")
    parser.add_argument("--fallback", action="store_true", help="use the 0.5B fallback model")
    parser.add_argument("--train-probe", action="store_true", help="also run a LoRA fwd+bwd memory probe")
    parser.add_argument("--skip-generation", action="store_true", help="skip the generation smoke test")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 0 gate -- QLoRA text-to-SQL environment verification")
    print("=" * 72)

    report = Report()
    config = load_config()

    if args.model:
        model_id = args.model
    elif args.fallback:
        model_id = config["model"]["fallback_id"]
    else:
        model_id = config["model"]["id"]

    check_interpreter(report)
    torch, gpu_info = check_torch_and_gpu(report)

    if torch is None or not gpu_info:
        return finish(report)
    if not check_bitsandbytes(report):
        return finish(report)

    model, tokenizer = load_model_4bit(report, config, model_id, gpu_info["supports_bf16"])
    if model is None:
        return finish(report)

    run_forward_pass(report, model, tokenizer)

    if not args.skip_generation:
        run_generation_smoke_test(report, model, tokenizer)
    if args.train_probe:
        run_train_probe(report, model, tokenizer)

    return finish(report)


def finish(report: Report) -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)

    failed = [c.name for c in report.checks if not c.passed]

    print("\n" + "=" * 72)
    if report.ok:
        print("PHASE 0: PASS -- environment verified, proceed to Phase 1")
    else:
        print(f"PHASE 0: FAIL -- {len(failed)} check(s) failed: {', '.join(failed)}")
        print("Do not proceed to Phase 1. See docs/SETUP.md for remediation.")
    print(f"Report written to {REPORT_PATH}")
    print("=" * 72)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
