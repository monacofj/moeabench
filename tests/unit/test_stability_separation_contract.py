# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #108: guard independent numeric and performance certification dispatch.

Migrates the superseded CLI contract without weakening its original safety
requirements: no implicit cross-trigger, no missing references, and fail closed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from moeabench import certification as cert


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / "test.py"
    spec = importlib.util.spec_from_file_location("_certification_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numeric_dispatch_never_calls_performance(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(sys, "argv", ["test.py", "--numeric-stability", "--basic"])
    monkeypatch.setattr(runner, "_run_pytest", lambda label, args, marker: (
        calls.append((label, marker)) or True,
        {"collected": 1, "passed": 1, "failed": 0, "errors": 0,
         "skipped": 0, "warnings": 0},
        0.0,
    ))
    monkeypatch.setattr(runner, "_print_final_summary", lambda rows: None)
    monkeypatch.setattr(cert, "_measure", lambda *a, **k: pytest.fail(
        "Performance measurement invoked implicitly by numerical stability"
    ))
    with pytest.raises(SystemExit) as e:
        runner.main()
    assert e.value.code == 0
    assert calls == [("STABILITY (BASIC)", "scope_stability and (level_smoke or level_basic)")]


def test_old_public_aliases_are_not_silently_accepted(monkeypatch, runner):
    for args in (["--certify"], ["--performance-stability"], ["--stability"],
                 ["--performance-reference", "any.json"]):
        monkeypatch.setattr(sys, "argv", ["test.py", *args])
        with pytest.raises(SystemExit) as e:
            runner.main()
        assert e.value.code == 2


def test_deep_without_statistical_reference_fails_closed(monkeypatch, runner):
    monkeypatch.setattr(sys, "argv", ["test.py", "--numeric-stability", "--deep"])
    with pytest.raises(SystemExit) as e:
        runner.main()
    assert e.value.code == 2


def test_performance_certification_requires_explicit_reference_and_output(monkeypatch, runner):
    for args in (["--certify-performance"], ["--certify-performance", "--output", "out.json"],
                 ["--certify-performance", "--reference", "reference.json"]):
        monkeypatch.setattr(sys, "argv", ["test.py", *args])
        with pytest.raises(SystemExit) as e:
            runner.main()
        assert e.value.code == 2


def test_performance_dispatch_never_runs_numeric_suite(monkeypatch, runner, tmp_path):
    out = tmp_path / "out.json"
    calls = []
    monkeypatch.setattr(sys, "argv", ["test.py", "--certify-performance",
                                      "--reference", "approved.json", "--output", str(out)])
    monkeypatch.setattr(runner, "_run_pytest", lambda *a, **k: pytest.fail(
        "Numerical stability suite invoked by performance certification"
    ))
    monkeypatch.setattr(runner, "_certify_environment", lambda: pytest.fail(
        "Numeric certification invoked by performance certification"
    ))
    monkeypatch.setattr(cert, "certify_performance", lambda source, destination: (
        calls.append((source, destination)) or
        (out, {"status": "CERTIFIED"})
    ))
    with pytest.raises(SystemExit) as e:
        runner.main()
    assert e.value.code == 0
    assert calls == [("approved.json", str(out))]


def test_statistically_insufficient_evidence_blocks_release(monkeypatch, runner, tmp_path):
    monkeypatch.setattr(sys, "argv", ["test.py", "--certify-performance",
                                      "--reference", "approved.json",
                                      "--output", str(tmp_path / "out.json")])
    monkeypatch.setattr(cert, "certify_performance", lambda *a: (
        tmp_path / "out.json", {"status": "NOT_CERTIFIED"}
    ))
    with pytest.raises(SystemExit) as e:
        runner.main()
    assert e.value.code == 1


def test_baseline_record_does_not_certify_or_overwrite(monkeypatch, runner, tmp_path):
    destination = tmp_path / "candidate.json"
    calls = []
    monkeypatch.setattr(sys, "argv", ["test.py", "--record-performance-baseline",
                                      "--output", str(destination)])
    monkeypatch.setattr(runner, "_certify_environment", lambda: pytest.fail(
        "Baseline creation must not launch numeric certification"
    ))
    monkeypatch.setattr(cert, "create_performance_baseline", lambda path: (
        calls.append(path) or destination
    ))
    with pytest.raises(SystemExit) as e:
        runner.main()
    assert e.value.code == 0
    assert calls == [str(destination)]


def test_numeric_certification_failure_never_writes_manifest(monkeypatch, runner):
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=1))
    assert runner._certify_environment() == 1


def test_numeric_certification_does_not_record_performance(monkeypatch, runner, tmp_path):
    monkeypatch.setattr(sys, "argv", ["test.py", "--certify-numeric"])
    calls = []
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(runner, "_certify_environment", lambda: 0)
    monkeypatch.setattr(cert, "register_numeric_history", lambda **kw: (
        calls.append(kw) or tmp_path / "numeric-certificate.json"
    ))
    monkeypatch.setattr(cert, "create_performance_baseline", lambda *a, **k: pytest.fail(
        "Numeric certification unexpectedly recorded performance baseline"
    ))
    with pytest.raises(SystemExit) as e:
        runner.main()
    assert e.value.code == 0
    assert len(calls) == 1
    assert calls[0]["counts"]["status"] == "PASSED"
