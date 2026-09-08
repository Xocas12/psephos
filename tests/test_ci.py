"""The CI workflow must keep promising what the project promises.

A workflow that quietly drops a command, a Python version or the guardrail turns every later
failure into a surprise. These tests read the workflow file back and check the promises are
still written down.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow_text() -> str:
    assert WORKFLOW.is_file(), ".github/workflows/ci.yml is missing"
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_exists_and_runs_on_push_and_pr():
    trigger_block = _workflow_text().split("jobs:", 1)[0]
    assert "push" in trigger_block, "CI no longer runs on push"
    assert "pull_request" in trigger_block, "CI no longer runs on pull requests"


def test_workflow_runs_the_suite_and_the_linter():
    text = _workflow_text()
    for command in (
        "uv sync --all-extras",
        "uv run pytest -q",
        "uv run ruff check .",
        "uv run ruff format --check .",
    ):
        assert command in text, f"CI no longer runs: {command}"


def test_workflow_tests_every_python_version_the_matrix_names():
    text = _workflow_text()
    matrices = [line for line in text.splitlines() if "python-version:" in line and "3.10" in line]
    assert matrices, "CI no longer tests Python 3.10"
    for version in ("3.10", "3.11", "3.12", "3.13"):
        assert version in matrices[0], f"CI no longer tests Python {version}"


def test_workflow_names_the_guardrail_test_and_asserts_the_invariant_directly():
    text = _workflow_text()
    named = "tests/test_audit.py::test_a_flagged_finding_cannot_omit_confounds"
    assert named in text, "CI no longer names the guardrail test"
    # The direct assertion is the backstop: it must survive even if the test is renamed or
    # deleted, so the workflow has to keep constructing the offending Finding itself.
    assert 'Finding(check="x", title="t", flag=flag)' in text, (
        "CI no longer asserts the guardrail invariant directly"
    )
    # Both flagged levels must be exercised. A single try block would test only the first,
    # because the first call raises, so the workflow loops over them instead.
    assert "for flag in (Flag.STRONG, Flag.NOTABLE):" in text, (
        "CI no longer checks every flagged level"
    )
