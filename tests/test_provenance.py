"""Tests for provenance stamping.

`capture()` is what makes a rerun's numbers attributable to a specific commit
and library set rather than guessed at; these tests only need to confirm the
schema is right and that git and package lookups behave sanely, not that any
particular commit or version is present.
"""

import re
from pathlib import Path

from lora_text_to_sql.provenance import _package_versions, capture

REPO_ROOT = Path(__file__).resolve().parents[1]

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class TestCapture:
    def test_returns_the_expected_schema(self):
        result = capture(REPO_ROOT)
        assert set(result) == {"git_sha", "git_dirty", "python_version", "packages"}

    def test_git_sha_is_a_real_commit_hash(self):
        """This test file lives inside a real git checkout, so a SHA must
        resolve -- this is not run against a tarball export."""
        result = capture(REPO_ROOT)
        assert result["git_sha"] is not None
        assert _SHA_RE.match(result["git_sha"])

    def test_git_dirty_is_a_bool_when_git_succeeds(self):
        result = capture(REPO_ROOT)
        assert isinstance(result["git_dirty"], bool)

    def test_missing_repo_gives_none_git_fields_not_a_crash(self, tmp_path):
        """A directory with no .git must degrade gracefully, since a report
        should still be written even when git metadata is unavailable."""
        result = capture(tmp_path)
        assert result["git_sha"] is None
        assert result["git_dirty"] is None

    def test_python_version_matches_running_interpreter(self):
        import sys

        result = capture(REPO_ROOT)
        assert result["python_version"] == "%d.%d.%d" % sys.version_info[:3]


class TestPackageVersions:
    def test_tracked_packages_are_all_present_as_keys(self):
        versions = _package_versions()
        assert set(versions) == {
            "torch", "transformers", "peft", "trl", "bitsandbytes",
            "accelerate", "datasets",
        }

    def test_not_installed_reports_none_not_a_crash(self):
        """Whether or not the GPU stack happens to be installed in the
        environment running this test, every value must be a string or None,
        never raise -- this is the whole point of capturing provenance in CI,
        where these packages are deliberately not installed."""
        versions = _package_versions()
        for name, value in versions.items():
            assert value is None or isinstance(value, str), f"{name}: {value!r}"
