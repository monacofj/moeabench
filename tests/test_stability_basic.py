# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Basic numerical stability suite for analytical outputs and solver fronts."""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import moeabench as mb
from moeabench.diagnostics import auditor


DATA_JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration_reference_audit_v0.13.2.json")
TARGETS_JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration_reference_targets_v0.13.2.json")
FRONTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "stability_reference_fronts.json")
GT_JSON_PATH = os.path.join(os.path.dirname(__file__), "stability_reference_gt.json")

with open(DATA_JSON_PATH, "r") as f:
    CALIBRATION_REF_DATA = json.load(f)

with open(TARGETS_JSON_PATH, "r") as f:
    CALIBRATION_REF_TARGETS = json.load(f)

with open(FRONTS_JSON_PATH, "r") as f:
    STABILITY_FRONTS = json.load(f)

with open(GT_JSON_PATH, "r") as f:
    STABILITY_GT = json.load(f)

PROBLEMS = sorted(list(CALIBRATION_REF_DATA["problems"].keys()))
ALGORITHMS = ["NSGA2", "MOEAD", "NSGA3"]
GT_PROBLEMS = [
    "DTLZ8",
    "DTLZ9",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DPF5",
]


def _latest_audit_13x(reports_dir: str) -> str:
    best = None
    best_tuple = None
    for name in os.listdir(reports_dir):
        m = re.match(r"audit_v0\.13\.(\d+)\.json$", name)
        if not m:
            continue
        v = int(m.group(1))
        if best_tuple is None or v > best_tuple:
            best_tuple = v
            best = os.path.join(reports_dir, name)
    if not best:
        raise AssertionError("No audit_v0.13.x.json found in calibration/reports")
    return best


def _iter_numeric(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_numeric(v, p)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            p = f"{prefix}[{i}]"
            yield from _iter_numeric(v, p)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield prefix, float(d)


def _build_problem(mop_name: str):
    mop_cls = getattr(mb.mops, mop_name)
    try:
        return mop_cls(M=3)
    except TypeError:
        return mop_cls(M=3, D=2)


def _sort_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    order = np.lexsort(arr.T[::-1])
    return arr[order]


def _gt_geometry_signature(problem_name: str) -> tuple[float, float]:
    p_data = CALIBRATION_REF_DATA["problems"][problem_name]
    expected = np.asarray(p_data["gt_points"], dtype=float)
    mop = _build_problem(problem_name)
    observed = np.asarray(mop.pf(n_points=len(expected)), dtype=float)
    forward = cdist(observed, expected).min(axis=1)
    backward = cdist(expected, observed).min(axis=1)
    return float(np.max(forward)), float(np.max(backward))


def test_audit_0140_matches_latest_013x():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reports_dir = os.path.join(repo_root, "calibration", "reports")
    audit_0140 = os.path.join(reports_dir, "audit_v0.14.0.json")
    audit_013x = _latest_audit_13x(reports_dir)

    assert os.path.exists(audit_0140), f"Missing promoted calibration audit: {audit_0140}"
    assert os.path.exists(audit_013x), f"Missing latest 0.13.x audit: {audit_013x}"

    with open(audit_0140, "r") as f:
        d140 = json.load(f)
    with open(audit_013x, "r") as f:
        d13x = json.load(f)

    p140 = d140.get("problems", {})
    p13x = d13x.get("problems", {})
    assert set(p140.keys()) == set(p13x.keys())

    for prob in sorted(p140.keys()):
        a140 = p140[prob].get("algorithms", {})
        a13x = p13x[prob].get("algorithms", {})
        assert set(a140.keys()) == set(a13x.keys())

        for alg in sorted(a140.keys()):
            n140 = dict(_iter_numeric(a140[alg].get("clinical", {}), "clinical"))
            n13x = dict(_iter_numeric(a13x[alg].get("clinical", {}), "clinical"))
            assert set(n140.keys()) == set(n13x.keys())
            for key in n140:
                assert n140[key] == pytest.approx(n13x[key], abs=1e-12)

            m140 = dict(_iter_numeric(a140[alg].get("stats", {}), "stats"))
            m13x = dict(_iter_numeric(a13x[alg].get("stats", {}), "stats"))
            assert set(m140.keys()) == set(m13x.keys())
            for key in m140:
                assert m140[key] == pytest.approx(m13x[key], abs=1e-12)


def test_calibration_artifact_consistency():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    report_alias = os.path.join(repo_root, "calibration", "audit_report.html")
    reports_dir = os.path.join(repo_root, "calibration", "reports")

    assert os.path.exists(report_alias)

    resolved_report = os.path.realpath(report_alias)
    m = re.search(r"audit_report_v([0-9]+\.[0-9]+\.[0-9]+)\.html$", resolved_report)
    assert m
    report_ver = m.group(1)
    expected_json = os.path.join(reports_dir, f"audit_v{report_ver}.json")
    assert os.path.exists(expected_json)

    from calibration.scripts.generate_visual_report import resolve_default_audit_json

    resolved_json = os.path.realpath(resolve_default_audit_json(repo_root))
    assert os.path.realpath(expected_json) == resolved_json


@pytest.mark.parametrize("problem_name", GT_PROBLEMS, ids=GT_PROBLEMS)
def test_gt_reference_reproducibility(problem_name):
    expected = STABILITY_GT["basic"][problem_name]
    forward_max, backward_max = _gt_geometry_signature(problem_name)
    assert forward_max <= expected["forward_max"] + 1e-12
    assert backward_max <= expected["backward_max"] + 1e-12


@pytest.mark.parametrize("problem_name", PROBLEMS, ids=PROBLEMS)
@pytest.mark.parametrize("alg_name", ALGORITHMS, ids=ALGORITHMS)
def test_metric_reference_reproducibility(problem_name, alg_name):
    p_data = CALIBRATION_REF_DATA["problems"][problem_name]
    gt_points = np.array(p_data["gt_points"])
    if alg_name not in p_data["algorithms"]:
        pytest.skip(f"Algorithm {alg_name} not found for {problem_name}")

    a_data = p_data["algorithms"][alg_name]
    final_front = np.array(a_data["final_front"])

    target_key = f"{problem_name}/{alg_name}"
    if target_key not in CALIBRATION_REF_TARGETS:
        pytest.skip(f"No calibration reference target for {target_key}")

    target_entry = CALIBRATION_REF_TARGETS[target_key]
    if target_entry.get("status") != "OK":
        pytest.skip(f"Calibration reference target invalid for {target_key}: {target_entry.get('error')}")

    report = auditor.audit(final_front, ground_truth=gt_points, problem=problem_name)
    q_audit = report.q_audit_res
    fair_audit = report.fair_audit_res

    assert q_audit is not None
    assert fair_audit is not None

    for key, target_val in target_entry["q_scores"].items():
        if key in q_audit.scores:
            computed_val = float(q_audit.scores[key].value)
            assert computed_val == pytest.approx(target_val, abs=1e-6, nan_ok=True)

    for key, target_val in target_entry["raw_metrics"].items():
        if key in fair_audit.metrics:
            computed_val = float(fair_audit.metrics[key].value)
            assert computed_val == pytest.approx(target_val, abs=1e-6, nan_ok=True)


@pytest.mark.parametrize(
    "entry",
    STABILITY_FRONTS["basic"],
    ids=lambda entry: f"{entry['algorithm']}x{entry['mop']}",
)
def test_front_reproducibility(entry):
    mop = _build_problem(entry["mop"])
    alg = getattr(mb.moeas, entry["algorithm"])(
        seed=entry["seed"],
        population=entry["population"],
        generations=entry["generations"],
    )
    exp = mb.experiment(mop=mop)
    exp.moea = alg
    exp.run(repeat=1, silent=True)

    observed = _sort_rows(np.asarray(exp.front()))
    expected = _sort_rows(np.asarray(entry["front"], dtype=float))
    assert observed.shape == expected.shape
    assert np.allclose(observed, expected, atol=1e-12, rtol=1e-8)
