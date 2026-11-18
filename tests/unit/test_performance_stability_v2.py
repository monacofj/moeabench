# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #98 v2 statistical protocol: additive, synthetic, independent checks.

No performance baseline is generated or certified by these unit tests.
Legacy v1 tests and all frozen numerical regressions are left unchanged.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from moeabench import performance_stability as ps
from moeabench import performance_stability_v2 as v2


def _case(id="case-a", sd=0.01, offsets=None):
    baseline = [0.40 + i * 0.001 for i in range(8)]
    return {
        "id": id,
        "mop": {"name": "DTLZ2", "kwargs": {"M": 2}},
        "moea": {"name": "NSGA2", "kwargs": {"population": 12, "generations": 2}},
        "metric": "igd",
        "reference_front": [[0.0, 1.0], [1.0, 0.0]],
        "seeds": list(range(101, 109)),
        "baseline_values": baseline,
        "max_relative_degradation": 0.01,
        "max_absolute_degradation": 0.05,
        "pilot_standard_deviation": sd,
        "sample_size_rationale": "Pilot paired differences, 80% power and familywise 5%.",
    }


def _document(cases=None):
    return {
        "schema": v2.SCHEMA_V2,
        "provenance": {
            "source_revision": "external-prior-release-placeholder",
            "reference_origin": "synthetic-test-only-not-an-accepted-baseline",
            "reviewed_by": "mock-reviewer-for-schema-test",
            "environment_fingerprint": "synthetic-environment-for-unit-tests",
            "measurement_protocol": "paired-fixed-seed-minimized-igd",
            "pilot_origin": "synthetic-sd-for-unit-tests",
        },
        "family_alpha": 0.05,
        "target_power": 0.8,
        "cases": [_case()] if cases is None else cases,
    }


def _assess(case, offsets, *, family=1):
    observed = [
        baseline + offset
        for baseline, offset in zip(case["baseline_values"], offsets)
    ]
    return v2.assess_case_v2(
        case, observed, family_alpha=0.05, family_cases=family,
        target_power=0.8,
    )


def test_v2_distinguishes_evidence_of_no_material_regression_and_inconclusive():
    case = _case()
    no_change = _assess(case, [0.0] * 8)
    assert no_change["status"] == "PASSED"
    assert no_change["ci_upper"] < no_change["practical_floor"]
    assert no_change["runs"] >= no_change["required_runs"]

    ambiguous = _assess(case, [0.00, 0.03, 0.07, 0.10, 0.02, 0.09, 0.04, 0.08])
    assert ambiguous["status"] == "INCONCLUSIVE"
    assert ambiguous["ci_lower"] < ambiguous["practical_floor"] < ambiguous["ci_upper"]
    assert v2.aggregate_v2([no_change, ambiguous]) == "INCONCLUSIVE"


def test_v2_detects_material_worsening_and_aggregate_precedence():
    case = _case()
    failed = _assess(case, [0.10] * 8)
    assert failed["status"] == "FAILED"
    assert failed["ci_lower"] > failed["practical_floor"]
    assert v2.aggregate_v2([_assess(case, [0.0] * 8), failed]) == "FAILED"


def test_v2_multiple_comparison_correction_is_more_conservative():
    case = _case()
    offsets = [0.00, 0.03, 0.07, 0.10, 0.02, 0.09, 0.04, 0.08]
    single = _assess(case, offsets, family=1)
    doubled = _assess(case, offsets, family=2)
    assert doubled["per_tail_alpha"] == pytest.approx(single["per_tail_alpha"] / 2)
    assert doubled["ci_upper"] > single["ci_upper"]
    assert doubled["ci_lower"] < single["ci_lower"]


def test_v2_reference_requires_family_policy_power_provenance_and_actual_seeds(tmp_path):
    good = _document()
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert ps.load_reference(path)["schema"] == v2.SCHEMA_V2
    for edited, expected_error in (
        (dict(good, family_alpha=1.0), "family_alpha"),
        (dict(good, target_power=0.65), "target_power"),
        (dict(good, cases=[_case(sd=0.10)]), "minimum"),
        (dict(good, cases=[dict(_case(), seeds=[101] * 8)]), "seeds"),
        (dict(good, cases=[dict(_case(), alpha=0.1)]), "family_alpha"),
        (dict(good, provenance=dict(good["provenance"], pilot_origin="")), "provenance"),
    ):
        path.write_text(json.dumps(edited), encoding="utf-8")
        with pytest.raises(ValueError, match=expected_error):
            ps.load_reference(path)


def test_v2_executes_only_declared_cases_and_preserves_inconclusive(monkeypatch, tmp_path):
    cases = [_case("case-a"), _case("case-b")]
    document = _document(cases)
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    invoked = []

    def fake_measure(case, seed):
        invoked.append((case["id"], seed))
        idx = case["seeds"].index(seed)
        changes = [0.00, 0.03, 0.07, 0.10, 0.02, 0.09, 0.04, 0.08]
        offset = 0 if case["id"] == "case-a" else changes[idx]
        return case["baseline_values"][idx] + offset

    monkeypatch.setattr(ps, "_measure_case", fake_measure)
    result = ps.verify_performance(path, show=False)
    assert result["status"] == "INCONCLUSIVE"
    assert len(invoked) == 16
    assert invoked == [
        (case["id"], seed) for case in cases for seed in case["seeds"]
    ]
    assert [r["status"] for r in result["cases"]] == ["PASSED", "INCONCLUSIVE"]


def test_v2_cli_exits_nonzero_for_inconclusive_without_numeric_suite(
        monkeypatch, tmp_path):
    document = _document()
    reference = tmp_path / "reviewed.json"
    reference.write_text(json.dumps(document), encoding="utf-8")
    original = ps._measure_case

    def ambiguous(case, seed):
        idx = case["seeds"].index(seed)
        delta = [0.00, 0.03, 0.07, 0.10, 0.02, 0.09, 0.04, 0.08][idx]
        return case["baseline_values"][idx] + delta

    monkeypatch.setattr(ps, "_measure_case", ambiguous)
    path = Path(__file__).resolve().parents[2] / "test.py"
    spec = importlib.util.spec_from_file_location("_test_runner_v2", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "_run_pytest", lambda *a, **k: pytest.fail(
        "Numeric-stability suite must not be triggered by performance diagnostics"
    ))
    monkeypatch.setattr(sys, "argv", [
        "test.py", "--performance-stability",
        "--performance-reference", str(reference),
    ])
    with pytest.raises(SystemExit) as ended:
        runner.main()
    assert ended.value.code == 1
    assert ps._measure_case is ambiguous
    monkeypatch.setattr(ps, "_measure_case", original)
