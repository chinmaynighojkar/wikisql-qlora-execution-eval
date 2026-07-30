"""Tests for the extracted, testable stages of run_eval.main().

Before this decomposition, run_eval.main() was one 107-line function doing
argument parsing, generation, scoring, report assembly, and printing in
sequence -- nothing in it could be tested without a GPU and a downloaded
model. These stages are pure or file-only, so they can be tested without
either.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_run_eval():
    spec = importlib.util.spec_from_file_location(
        "run_eval", REPO_ROOT / "scripts" / "run_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_eval"] = module
    spec.loader.exec_module(module)
    return module


run_eval = load_run_eval()


class TestArgParser:
    def test_defaults(self):
        args = run_eval.build_arg_parser().parse_args([])
        assert args.name == "baseline"
        assert args.adapter is None
        assert args.limit is None
        assert args.batch_size == 8
        assert args.seed == run_eval.DEFAULT_SEED

    def test_overrides(self):
        args = run_eval.build_arg_parser().parse_args(
            ["--name", "finetuned", "--adapter", "models/lora-adapter", "--limit", "10"]
        )
        assert args.name == "finetuned"
        assert args.adapter == "models/lora-adapter"
        assert args.limit == 10


class TestLoadSavedPredictions:
    def test_loads_raw_output_field(self, tmp_path):
        path = tmp_path / "predictions.json"
        path.write_text(
            json.dumps([{"raw_output": "SELECT 1"}, {"raw_output": "SELECT 2"}]),
            encoding="utf-8",
        )
        outputs = run_eval.load_saved_predictions(str(path), n_records=2)
        assert outputs == ["SELECT 1", "SELECT 2"]

    def test_mismatched_count_exits(self, tmp_path):
        path = tmp_path / "predictions.json"
        path.write_text(json.dumps([{"raw_output": "SELECT 1"}]), encoding="utf-8")
        with pytest.raises(SystemExit, match="mismatch"):
            run_eval.load_saved_predictions(str(path), n_records=2)


class TestBuildReport:
    def _args(self, **overrides):
        defaults = {
            "name": "baseline", "adapter": None, "model": None,
            "max_new_tokens": 128,
        }
        defaults.update(overrides)
        return run_eval.argparse.Namespace(**defaults)

    def test_schema_and_provenance(self):
        records = [{"question": "q1"}, {"question": "q2"}]
        metrics = {"execution_accuracy": 0.5}
        report = run_eval.build_report(self._args(), records, 12.3, metrics)
        assert report["name"] == "baseline"
        assert report["n_records"] == 2
        assert report["generation_seconds"] == 12.3
        assert report["decoding"] == {"strategy": "greedy", "max_new_tokens": 128}
        assert report["metrics"] is metrics
        assert "provenance" in report and "git_sha" in report["provenance"]

    def test_carries_adapter_and_model_through(self):
        records = [{"question": "q1"}]
        report = run_eval.build_report(
            self._args(adapter="models/lora-adapter", model="Qwen/Qwen2.5-1.5B-Instruct"),
            records, 0.0, {},
        )
        assert report["adapter"] == "models/lora-adapter"
        assert report["model"] == "Qwen/Qwen2.5-1.5B-Instruct"


class TestWriteReport:
    def test_writes_both_files_with_expected_names(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_eval, "REPORTS_DIR", tmp_path)
        report = {"name": "baseline", "metrics": {"execution_accuracy": 0.5}}

        class FakeResult:
            def as_dict(self):
                return {"question": "q1"}

        metrics_path, predictions_path = run_eval.write_report(
            "baseline", report, [FakeResult(), FakeResult()]
        )
        assert metrics_path == tmp_path / "baseline_metrics.json"
        assert predictions_path == tmp_path / "baseline_predictions.json"
        assert json.loads(metrics_path.read_text())["name"] == "baseline"
        assert len(json.loads(predictions_path.read_text())) == 2

    def test_creates_reports_dir_if_missing(self, tmp_path, monkeypatch):
        target = tmp_path / "nested" / "reports"
        monkeypatch.setattr(run_eval, "REPORTS_DIR", target)
        run_eval.write_report("x", {"name": "x"}, [])
        assert target.exists()
