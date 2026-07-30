"""Provenance stamping for report artefacts.

The model revision is pinned to a commit SHA (D-004) precisely so the base
model cannot drift between runs. Reports recorded the model, decoding
strategy and record counts, but not the code or library versions that
produced them -- so a number that changes on a rerun months from now could
not be attributed to a code change, a library upgrade, or a different GPU.
This closes the same gap the model pin closes, for the other half of what
"reproducible" requires.
"""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# The packages whose exact versions D-005 already pins in pyproject.toml,
# because those are the ones a mid-project upgrade could shift results for
# reasons unrelated to fine-tuning.
_TRACKED_PACKAGES = (
    "torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _run_git(args: list[str], repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git checkout, or git is unavailable. A report should still be
        # written; it is simply missing the one field that needed git.
        return None


def capture(repo_root: Path) -> dict[str, Any]:
    """Snapshot the code and library state that produced a report.

    Call once per script run and embed the result in every report the run
    writes, so a number that differs on a future rerun can be attributed to a
    specific commit and a specific set of library versions rather than
    guessed at.
    """
    sha = _run_git(["rev-parse", "HEAD"], repo_root)
    dirty_output = _run_git(["status", "--porcelain"], repo_root)
    return {
        "git_sha": sha,
        "git_dirty": bool(dirty_output) if dirty_output is not None else None,
        "python_version": platform.python_version(),
        "packages": _package_versions(),
    }
