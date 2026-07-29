"""Regression tests for the Phase 0 verification script.

These exist because of a real failure. The GPU-present branch of
`check_torch_and_gpu` had never been executed before it ran on the target
machine: the development sandbox has no CUDA device, so every test run
returned early at the `torch_import` check and the success path was dead code
that looked tested. It crashed on first contact with a real GPU.

The fix was twofold -- rename the colliding payload key, and make the
`Report.add` positional parameters positional-only -- and these tests stub a
CUDA device so the success path is exercised without one.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_phase0():
    spec = importlib.util.spec_from_file_location(
        "phase0_verify_env", REPO_ROOT / "scripts" / "phase0_verify_env.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register before executing: @dataclass resolves field types via
    # sys.modules[cls.__module__], which is None until the module is present.
    sys.modules["phase0_verify_env"] = module
    spec.loader.exec_module(module)
    return module


phase0 = load_phase0()


# --------------------------------------------------------------------------
# The exact bug
# --------------------------------------------------------------------------


class TestReportAdd:
    def test_payload_key_named_name_does_not_collide(self):
        """`report.add("x", True, "y", **{"name": ...})` raised
        `TypeError: got multiple values for argument 'name'`."""
        report = phase0.Report()
        report.add("cuda_available", True, "detail", **{"name": "RTX 3050"})
        assert report.checks[0].name == "cuda_available"
        assert report.checks[0].data["name"] == "RTX 3050"

    @pytest.mark.parametrize("key", ["name", "passed", "detail"])
    def test_no_parameter_name_can_be_shadowed(self, key):
        report = phase0.Report()
        report.add("check", True, "detail", **{key: "payload"})
        assert report.checks[0].data[key] == "payload"

    def test_failure_is_recorded(self):
        report = phase0.Report()
        report.add("a", True, "ok")
        report.add("b", False, "bad")
        assert not report.ok
        assert report.to_dict()["overall_result"] == "FAIL"


# --------------------------------------------------------------------------
# Stubbed CUDA device
# --------------------------------------------------------------------------


def make_fake_torch(*, cuda_build="13.0", available=True, major=8, minor=6,
                    total=4 * 1024**3, free=3.9 * 1024**3):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.13.0+cu130"
    torch.version = types.SimpleNamespace(cuda=cuda_build)

    props = types.SimpleNamespace(
        name="NVIDIA GeForce RTX 3050 Laptop GPU", major=major, minor=minor
    )
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        current_device=lambda: 0,
        get_device_properties=lambda i: props,
        mem_get_info=lambda i=0: (int(free), int(total)),
    )
    return torch


@pytest.fixture
def fake_torch(monkeypatch):
    def install(**kwargs):
        module = make_fake_torch(**kwargs)
        monkeypatch.setitem(sys.modules, "torch", module)
        return module

    return install


class TestEveryCallSiteIsCompatible:
    """Making `add`'s first three parameters positional-only broke seven call
    sites that passed `passed=` / `detail=` as keywords -- and the crash only
    surfaced on the target machine, because no test executed those functions.

    The static check below covers every call site in the file, including the
    ones inside GPU-only branches that cannot be executed here at all.
    """

    def test_no_call_site_uses_positional_only_names_as_keywords(self):
        import ast

        source = (REPO_ROOT / "scripts" / "phase0_verify_env.py").read_text(encoding="utf-8")
        reserved = {"name", "passed", "detail"}
        offenders = []

        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "add"):
                continue
            for keyword in node.keywords:
                if keyword.arg in reserved:
                    offenders.append((node.lineno, keyword.arg))

        assert not offenders, (
            "report.add() call sites passing a positional-only parameter by "
            f"keyword: {offenders}"
        )

    def test_every_call_site_supplies_three_positional_arguments(self):
        import ast

        source = (REPO_ROOT / "scripts" / "phase0_verify_env.py").read_text(encoding="utf-8")
        offenders = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "add"):
                continue
            if len(node.args) != 3:
                offenders.append((node.lineno, len(node.args)))
        assert not offenders, f"report.add() called with != 3 positional args: {offenders}"


class TestCheckInterpreter:
    """`check_interpreter` needs no GPU, so there was never an excuse for it
    to be untested -- and it was the first thing to crash."""

    def test_runs_and_records_both_checks(self):
        report = phase0.Report()
        phase0.check_interpreter(report)
        names = [c.name for c in report.checks]
        assert names == ["python_version", "platform"]
        assert report.ok

    def test_records_the_running_interpreter_version(self):
        report = phase0.Report()
        phase0.check_interpreter(report)
        assert report.checks[0].data["version"].startswith(
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )


class TestCheckTorchAndGpu:
    def test_success_path_on_an_ampere_card(self, fake_torch):
        """The path that crashed on the RTX 3050."""
        fake_torch()
        report = phase0.Report()
        torch, gpu_info = phase0.check_torch_and_gpu(report)

        assert torch is not None
        assert report.ok
        assert gpu_info["device_name"] == "NVIDIA GeForce RTX 3050 Laptop GPU"
        assert gpu_info["compute_capability"] == "8.6"
        assert gpu_info["total_vram_gib"] == 4.0
        assert gpu_info["supports_bf16"] is True

    def test_cpu_only_build_is_caught(self, fake_torch):
        """The other real failure from setup: torch 2.13.0+cpu installed as a
        transitive dependency, where `torch.version.cuda` is None."""
        fake_torch(cuda_build=None)
        report = phase0.Report()
        torch, gpu_info = phase0.check_torch_and_gpu(report)

        assert not report.ok
        assert gpu_info == {}
        assert any(c.name == "torch_cuda_build" and not c.passed for c in report.checks)

    def test_no_visible_device_is_caught(self, fake_torch):
        fake_torch(available=False)
        report = phase0.Report()
        _, gpu_info = phase0.check_torch_and_gpu(report)
        assert not report.ok and gpu_info == {}

    def test_pre_ampere_card_falls_back_to_fp16(self, fake_torch):
        """bf16 needs compute capability >= 8.0; a Turing card must not block
        the run, only switch the compute dtype."""
        fake_torch(major=7, minor=5)
        report = phase0.Report()
        _, gpu_info = phase0.check_torch_and_gpu(report)
        assert gpu_info["supports_bf16"] is False
        assert report.ok
