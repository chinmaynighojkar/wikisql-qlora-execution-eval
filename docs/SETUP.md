# Setup — Phase 0

Target machine for this project: Windows, Python 3.10, NVIDIA RTX 3050 Laptop
(4 GB VRAM, Ampere / sm_86).

The goal of Phase 0 is narrow and specific: **prove that Qwen2.5-1.5B-Instruct
loads in 4-bit NF4 and runs a forward pass without exhausting 4 GB of VRAM.**
Nothing else gets built until that passes, because every later phase depends
on it.

---

## 1. Create the virtual environment

From `C:\Projects\lora-text-to-sql`:

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Confirm you are actually on 3.10 before continuing — this is the most common
setup mistake when several Python versions are installed:

```bat
python --version
```

If `py -3.10` is not recognised, 3.10 is not installed. Install it from
python.org and re-run. Do not substitute a newer Python: newer versions on
this machine have lacked wheel support for parts of this stack.

## 2. Check the NVIDIA driver

```bat
nvidia-smi
```

Note the **CUDA Version** in the top-right of the output. That is the maximum
CUDA runtime the driver supports, and it must be **>= 12.6** for step 3. If it
is lower, update the NVIDIA driver first.

## 3. Install PyTorch with CUDA

`pip install torch` from PyPI installs the **CPU-only** build on Windows, and
bitsandbytes 4-bit kernels cannot run on it.

> **This trap fires even if you never type `pip install torch`.** torch is a
> transitive dependency of transformers, peft, trl and accelerate, so running
> `pip install -r requirements.txt` on its own will resolve torch from PyPI
> and install the CPU wheel. The failure is silent — every package imports
> fine, and only the 4-bit model load fails, much later and with a confusing
> error. This happened once during setup here, which is why `requirements.txt`
> now pins `torch==2.13.0+cu130` with an `--extra-index-url` rather than
> omitting torch: an omitted dependency is resolved by pip, a pinned local
> version is not.

With the pin in place, step 4 installs the correct build on its own. To do it
explicitly first:

```bat
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

Match the index to the driver's CUDA version from step 2 — `cu130` for CUDA
13.x, `cu126` for CUDA 12.6+. Both publish Python 3.10 Windows wheels; the
cu130 wheel is also ~650 MB smaller (1,827 MB vs 2,474 MB).

**On a slow or unreliable connection**, pip cannot resume a partial wheel — it
restarts from zero and fails the hash check on truncated bytes. Download with
a resumable client instead, verify, then install the local file:

```powershell
$file = "torch-2.13.0+cu130-cp310-cp310-win_amd64.whl"
$url  = "https://download.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp310-cp310-win_amd64.whl"
$want = "e85d18b0c51744b25fab85dbf590a4f00644da432af0e9d03c62acbd2f96ea94"
for ($i=1; $i -le 100; $i++) {
    curl.exe -L -C - --retry 5 --retry-delay 5 -o $file $url
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep 5
}
(Get-FileHash ".\$file" -Algorithm SHA256).Hash -eq $want.ToUpper()
pip install ".\$file"
```

Three things that each cost a retry when this was first done:

- **Keep the canonical wheel filename.** pip parses it for package metadata
  (`name-version-pythontag-abitag-platform.whl`). Saving it as `torch.whl`
  fails with `Invalid wheel filename (wrong number of parts): 'torch'`. The
  `%2B` in the URL is an encoded `+`, so the file on disk uses `+`.
- **`curl.exe`, not `curl`.** The bare name is a PowerShell alias for
  `Invoke-WebRequest`, which does not support `-C -` resume.
- **A `THESE PACKAGES DO NOT MATCH THE HASHES` error is almost always a
  truncated download,** not tampering — especially if throughput collapsed
  mid-transfer. Verify against the hash published on the index (as above)
  rather than disabling the check.

Verify the CUDA build took — do this before anything else, because every
later failure looks different and more confusing than this one:

```bat
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected: a version string ending in `+cu130` (or `+cu126`), a CUDA version,
and `True`. If it prints `+cpu` and `None`, the CPU wheel is installed —
`pip uninstall -y torch` and repeat this step. Checking
`.venv\Lib\site-packages\torch\version.py` directly is equally conclusive:
`cuda: Optional[str] = None` means CPU-only.

## 4. Install the rest

```bat
pip install -r requirements.txt
```

## 5. Run the Phase 0 gate

```bat
python scripts\phase0_verify_env.py
```

Optionally add `--train-probe` to also get an early read on training-time
memory (one LoRA forward+backward). That probe is *not* part of the Phase 0
pass criteria — see the note in the script — but it de-risks Phase 3 cheaply.

```bat
python scripts\phase0_verify_env.py --train-probe
```

The first run downloads roughly 3 GB of model weights from the Hugging Face
Hub and will take a few minutes. Subsequent runs load from the local cache.

**Pass criteria:** the script prints `PHASE 0: PASS`, exits `0`, and writes
`reports/phase0_env_report.json` with real measured VRAM figures.

---

## What the gate actually checks

Ordered so the cheapest and most likely failure comes first:

| # | Check | Why it exists |
|---|-------|---------------|
| 1 | Python version and platform | Records the environment in the report for reproducibility. |
| 2 | torch CUDA build + device properties | Catches the CPU-wheel mistake from step 3 before wasting a model download. |
| 3 | bitsandbytes NF4 kernel round-trip on GPU | The main platform risk. Importing bitsandbytes can succeed while the compiled CUDA backend is broken, so the check forces a real quantise/dequantise on the GPU rather than trusting the import. |
| 4 | 4-bit model load, pinned to GPU 0 | The core fit test. |
| 5 | No-CPU-offload assertion | See "false passes" below. |
| 6 | Forward pass + loss plausibility | Confirms the quantised weights produce sane outputs, not just that they fit. |
| 7 | VRAM headroom, device-wide and allocator-level | The number that decides whether Phase 3 is feasible. |
| 8 | Generation smoke test | Phase 2 needs `generate()` to work; failing here now is cheaper than failing there later. |

## Two false passes this is built to avoid

Both of these would make the script report success while proving nothing —
which is worse than a clean failure, because the project would be built on top
of it.

**Silent CPU offload.** The obvious way to load a model is
`device_map="auto"`. Under memory pressure, accelerate quietly moves layers to
CPU RAM instead of raising OOM. The script would pass, and the "it fits in
4 GB" claim would be false. This project pins `device_map={"": 0}` (set in
`configs/model.yaml`) and then walks every parameter asserting `device.type ==
"cuda"`.

**Windows CUDA sysmem fallback.** Recent NVIDIA drivers on Windows can spill
VRAM into system RAM rather than raising an out-of-memory error. Training
still "works" but runs an order of magnitude slower. Because of this, the
report records both `peak_torch_allocated_gib` (torch's own allocator, tensors
only) and `device_wide_used_gib` (from `cudaMemGetInfo`, which includes the
CUDA context and fragmentation). If the device-wide figure sits at or near the
card's total while throughput is poor, sysmem fallback is the likely cause. It
can be disabled per-application in NVIDIA Control Panel → Manage 3D Settings →
CUDA - Sysmem Fallback Policy → "Prefer No Sysmem Fallback".

---

## Fallbacks, in the order they should be tried

These are **decisions to record, not workarounds to apply silently.** If any
is used, add an entry to [DECISIONS.md](DECISIONS.md) and note it in the
README, following the same convention as the sibling `heart-disease-mlops`
compliance work.

### Fallback A — bitsandbytes fails on native Windows → WSL2

Trigger: check 3 fails (`bitsandbytes_import` or `bitsandbytes_nf4_kernel`).

bitsandbytes now ships an official `win_amd64` wheel, so native Windows is the
expected path and this fallback may not be needed. It is documented in advance
because the plan identified it as the top platform risk.

```bash
wsl --install -d Ubuntu-22.04     # from an elevated PowerShell, then reboot
```

Inside WSL2 the NVIDIA driver is provided by the Windows host — do **not**
install an NVIDIA driver inside the Linux guest:

```bash
sudo apt update && sudo apt install -y python3.10-venv
cd /mnt/c/Projects/lora-text-to-sql
python3.10 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python scripts/phase0_verify_env.py
```

Cost of this fallback: WSL2 reserves host RAM, and reading the dataset across
`/mnt/c` is slower than a native Linux filesystem. If it is used, move
`data/` inside the WSL2 filesystem.

### Fallback B — 1.5B does not fit → Qwen2.5-0.5B-Instruct

Trigger: check 4 raises OOM, or check 7 leaves too little headroom for
training.

```bat
python scripts\phase0_verify_env.py --fallback
```

This is a real scope reduction, not a neutral swap: a 0.5B model will produce
weaker absolute SQL accuracy. The *methodology* — execution-match before/after
with identical eval code — is unaffected, and that is what the project is
demonstrating. Record the reason and the measured numbers that forced it.

### Fallback C — inference fits but training does not

Trigger: `--train-probe` OOMs while the plain forward pass passed.

Options, cheapest first: reduce LoRA rank 16 → 8; restrict `target_modules` to
attention projections only (drop the MLP `gate/up/down_proj`); shorten
`max_seq_length`; then fall back to the 0.5B model. Phase 3 is where this gets
settled properly.

---

## Reporting a Phase 0 result

Paste the full console output, or the contents of
`reports/phase0_env_report.json`. The numbers that matter for the go/no-go
decision on Phase 1:

- `cuda_available.total_vram_gib` — what the card actually reports
- `model_load_4bit.weights_vram_gib_device_wide` — cost of the quantised weights
- `vram_headroom.device_wide_used_gib` / `device_wide_free_gib` — the headroom
- `forward_pass_loss_sane.loss` — sanity of the quantised weights
- `train_probe.peak_torch_allocated_gib` — if `--train-probe` was used
