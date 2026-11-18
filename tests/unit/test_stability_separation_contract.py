# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New contracts for explicit numerical/statistical test separation.

Historical regression tests, fixtures and benchmark targets are not altered.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from moeabench import performance_stability as ps


@pytest.fixture
def runner():
    script = Path(__file__).resolve().parents[2] / "test.py"
    spec = importlib.util.spec_from_file_location("_moeabench_test_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reference(tmp_path):
    case = {
        "id": "paired-igd",
        "mop": {"name": "DTLZ2", "kwargs": {"M": 2}},
        "moea": {"name": "NSGA2", "kwargs": {"population": 12, "generations": 2}},
        "metric": "igd",
        "reference_front": [[0.0, 1.0], [1.0, 0.0]],
        "seeds": [11, 12, 13],
        "baseline_values": [0.1, 0.11, 0.12],
        "max_relative_degradation": 0.1,
        "max_absolute_degradation": 0.02,
        "alpha": 0.05,
    }
    document = {
        "schema": ps.SCHEMA,
        "provenance": {
            "source_revision": "prior-release-sha",
            "reference_origin": "separate, reviewed historical result",
            "reviewed_by": "scientific-maintainer",
        },
        "cases": [case],
    }
    path = tmp_path / "performance.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def test_performance_reference_missing_or_unreviewed_fails_closed(tmp_path, reference):
    with pytest.raises(FileNotFoundError):
        ps.load_reference(tmp_path / "missing.json")
    path, document = reference
    document["provenance"]["reviewed_by"] = ""
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        ps.load_reference(path)


def test_performance_reference_rejects_seed_mismatch_and_duplicate_ids(reference):
    path, document = reference
    document["cases"][0]["seeds"] = [11, 11, 12]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="seeds"):
        ps.load_reference(path)
    document["cases"][0]["seeds"] = [11, 12, 13]
    document["cases"].append(document["cases"][0].copy())
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        ps.load_reference(path)


def test_independent_paired_statistical_decision(reference):
    case = reference[1]["cases"][0]
    assert ps.assess_case(case, [0.1, 0.11, 0.12])["status"] == "PASSED"
    regression = ps.assess_case(case, [0.2, 0.21, 0.22])
    assert regression["status"] == "FAILED"
    assert regression["runs"] == 3
    assert regression["mean_delta"] == pytest.approx(0.1)
    assert ps.assess_case(case, [0.1, 0.13, 0.12])["status"] == "PASSED"


def test_performance_runs_only_explicit_cases_and_seeds(monkeypatch, reference):
    path, _ = reference
    called = []
    def fake_measure(case, seed):
        called.append((case["id"], seed))
        return 0.1 + 0.01 * (seed - 11)

    monkeypatch.setattr(ps, "_measure_case", fake_measure)
    result = ps.verify_performance(path, show=False)
    assert result["status"] == "PASSED"
    assert called == [("paired-igd", 11), ("paired-igd", 12), ("paired-igd", 13)]


def test_numeric_dispatch_never_calls_performance(monkeypatch, runner):
    called = []
    monkeypatch.setattr(sys, "argv", ["test.py", "--numeric-stability", "--basic"])
    monkeypatch.setattr(runner, "_run_pytest", lambda label, args, marker: (
        called.append((label, marker)) or True,
        {"collected": 1, "passed": 1, "failed": 0, "errors": 0, "skipped": 0, "warnings": 0},
        0.0,
    ))
    monkeypatch.setattr(runner, "_print_final_summary", lambda rows: None)
    monkeypatch.setattr(ps, "verify_performance", lambda *a, **k: pytest.fail(
        "Performance verification was called implicitly"
    ))
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 0
    assert called == [("STABILITY (BASIC)", "scope_stability and (level_smoke or level_basic)")]


def test_obsolete_deep_is_not_reported_as_success(monkeypatch, runner):
    monkeypatch.setattr(sys, "argv", ["test.py", "--stability", "--deep"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_performance_requires_explicit_reference(monkeypatch, runner):
    monkeypatch.setattr(sys, "argv", ["test.py", "--performance-stability"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_performance_dispatch_does_not_start_pytest_or_numeric_suite(monkeypatch, runner, reference):
    monkeypatch.setattr(sys, "argv", [
        "test.py", "--performance-stability", "--performance-reference", str(reference[0])
    ])
    monkeypatch.setattr(runner, "_run_pytest", lambda *a, **k: pytest.fail(
        "Numerical stability invoked by performance check"
    ))
    monkeypatch.setattr(ps, "verify_performance", lambda path: {
        "status": "PASSED", "source": path, "cases": []
    })
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 0


def test_certification_failure_never_writes_manifest(monkeypatch, runner):
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd: SimpleNamespace(returncode=1))
    assert runner._certify_environment() == 1
