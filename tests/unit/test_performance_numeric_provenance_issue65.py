# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract checks for independent source-bound certification events."""

from pathlib import Path


def test_performance_workflow_accepts_both_authorized_numeric_event_types():
    """A numeric certificate may come from a verified push or issue command.

    A valid explicit numeric issue_comment run is not falsely rejected as
    a missing artifact. Revision, success, and artifact verification must
    remain mandatory regardless of the triggering event.
    """
    root = Path(__file__).resolve().parents[2]
    workflow = (
        root / ".github/workflows/release-100-performance-certification.yml"
    ).read_text()
    assert "NUMERIC_EVENTS:" in workflow
    assert "'issue_comment,push'" in workflow
    assert 'run["event"] in set(os.environ["NUMERIC_EVENTS"].split(","))' in workflow
    assert 'run["conclusion"] == "success"' in workflow
    assert 'run["head_sha"] == sha' in workflow
    assert 'x["name"] == "release-100-numeric-candidate"' in workflow
    assert 'not x["expired"]' in workflow


def test_explicit_numeric_command_still_requires_exact_revision():
    root = Path(__file__).resolve().parents[2]
    workflow = (
        root / ".github/workflows/release-100-numeric-certification.yml"
    ).read_text()
    assert 'tokens[1]==actual==os.environ["GITHUB_SHA"]' in workflow
