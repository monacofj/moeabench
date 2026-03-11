# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
moeabench Regression Tier Testing Suite
=======================================

This tier contains strict numerical reproducibility tests against established 
Calibration Reference Data.

Data Source 1 (Points):  `tests/calibration_reference_audit_v0.13.2.json`
Data Source 2 (Targets): `tests/calibration_reference_targets_v0.13.2.json`

Logic: Verifies that the current diagnostic pipeline reproduces the reference
clinical scores and physical metrics for a variety of MOPs and MOEAs.
"""

import pytest
import json
import os
import re
import numpy as np
from moeabench.diagnostics import auditor, baselines
from moeabench import metrics

# Load Calibration Reference Data (Module Level)
DATA_JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration_reference_audit_v0.13.2.json")
TARGETS_JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration_reference_targets_v0.13.2.json")

print(f"Loading Calibration Reference Data from {DATA_JSON_PATH}...")
with open(DATA_JSON_PATH, "r") as f:
    CALIBRATION_REF_DATA = json.load(f)

print(f"Loading Calibration Reference Targets from {TARGETS_JSON_PATH}...")
with open(TARGETS_JSON_PATH, "r") as f:
    CALIBRATION_REF_TARGETS = json.load(f)

PROBLEMS = sorted(list(CALIBRATION_REF_DATA['problems'].keys()))
ALGORITHMS = ['NSGA2', 'MOEAD', 'NSGA3'] # Standard set in the reference file


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


def test_audit_0140_matches_latest_013x():
    """
    Regression gate for manual baseline promotion:
    v0.14.0 calibration audit must match the latest v0.13.x numerically.
    """
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
    assert set(p140.keys()) == set(p13x.keys()), "Problem set mismatch between v0.14.0 and latest v0.13.x"

    for prob in sorted(p140.keys()):
        a140 = p140[prob].get("algorithms", {})
        a13x = p13x[prob].get("algorithms", {})
        assert set(a140.keys()) == set(a13x.keys()), f"Algorithm set mismatch for problem {prob}"

        for alg in sorted(a140.keys()):
            c140 = a140[alg].get("clinical", {})
            c13x = a13x[alg].get("clinical", {})
            s140 = a140[alg].get("stats", {})
            s13x = a13x[alg].get("stats", {})

            n140 = dict(_iter_numeric(c140, "clinical"))
            n13x = dict(_iter_numeric(c13x, "clinical"))
            assert set(n140.keys()) == set(n13x.keys()), f"Clinical numeric keys mismatch for {prob}/{alg}"
            for k in n140:
                assert n140[k] == pytest.approx(n13x[k], abs=1e-12), f"Clinical drift at {prob}/{alg}::{k}"

            m140 = dict(_iter_numeric(s140, "stats"))
            m13x = dict(_iter_numeric(s13x, "stats"))
            assert set(m140.keys()) == set(m13x.keys()), f"Stats numeric keys mismatch for {prob}/{alg}"
            for k in m140:
                assert m140[k] == pytest.approx(m13x[k], abs=1e-12), f"Stats drift at {prob}/{alg}::{k}"


def test_calibration_artifact_consistency():
    """
    Verifies consistency among calibration artifacts used by report rendering:
    - `calibration/audit_report.html` -> `audit_report_vX.Y.Z.html`
    - corresponding `calibration/reports/audit_vX.Y.Z.json` exists
    - `generate_visual_report.resolve_default_audit_json()` resolves the same JSON
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    report_alias = os.path.join(repo_root, "calibration", "audit_report.html")
    reports_dir = os.path.join(repo_root, "calibration", "reports")

    assert os.path.exists(report_alias), f"Missing calibration report alias: {report_alias}"

    resolved_report = os.path.realpath(report_alias)
    m = re.search(r"audit_report_v([0-9]+\.[0-9]+\.[0-9]+)\.html$", resolved_report)
    assert m, (
        f"Unexpected report naming: {resolved_report}. "
        "Expected suffix audit_report_vX.Y.Z.html"
    )
    report_ver = m.group(1)
    expected_json = os.path.join(reports_dir, f"audit_v{report_ver}.json")
    assert os.path.exists(expected_json), (
        f"Report points to version {report_ver}, but corresponding JSON is missing: {expected_json}"
    )

    from calibration.scripts.generate_visual_report import resolve_default_audit_json
    resolved_json = os.path.realpath(resolve_default_audit_json(repo_root))
    assert os.path.realpath(expected_json) == resolved_json, (
        "Visual report resolver and report alias disagree.\n"
        f"Expected JSON: {expected_json}\n"
        f"Resolved JSON: {resolved_json}"
    )

@pytest.mark.parametrize("problem_name", PROBLEMS)
@pytest.mark.parametrize("alg_name", ALGORITHMS)
def test_calibration_reference_reproducibility(problem_name, alg_name):
    """
    Verifies that the current codebase reproduces the exact same diagnostic scores
    as the Calibration Reference snapshot for a given (Problem, Algorithm) pair.
    """
    # 1. Retrieve Reference Data
    try:
        p_data = CALIBRATION_REF_DATA['problems'][problem_name]
        gt_points = np.array(p_data['gt_points'])
        if alg_name not in p_data['algorithms']:
             pytest.skip(f"Algorithm {alg_name} not found for {problem_name}")
             
        a_data = p_data['algorithms'][alg_name]
        final_front = np.array(a_data['final_front'])
        
        # Key for targets
        target_key = f"{problem_name}/{alg_name}"
        if target_key not in CALIBRATION_REF_TARGETS:
             pytest.skip(f"No calibration reference target for {target_key}")
             
        target_entry = CALIBRATION_REF_TARGETS[target_key]
        if target_entry.get("status") != "OK":
             pytest.skip(f"Calibration reference target invalid for {target_key}: {target_entry.get('error')}")
             
        target_q_scores = target_entry["q_scores"]
        target_raw_metrics = target_entry["raw_metrics"]
        
    except KeyError as e:
        pytest.fail(f"Calibration Reference Data malformed for {problem_name}/{alg_name}: {e}")

    # 2. Run Audit (The "System Under Test")
    report = auditor.audit(final_front, ground_truth=gt_points, problem=problem_name)
    
    # 3. Verify Clinical Q-Scores
    q_audit = report.q_audit_res
    if q_audit is None:
        pytest.fail(f"Audit failed to produce Quality Scores for {problem_name}/{alg_name}")

    for key, target_val in target_q_scores.items():
        if key not in q_audit.scores:
            continue
        
        computed_val = float(q_audit.scores[key].value)
        assert computed_val == pytest.approx(target_val, abs=1e-6, nan_ok=True), \
            f"Regression in {key} for {problem_name}/{alg_name}"

    # 4. Verify Raw Fair Metrics
    fair_audit = report.fair_audit_res
    if fair_audit is None:
         pytest.fail(f"Audit failed to produce Fair Metrics for {problem_name}/{alg_name}")

    for key, target_val in target_raw_metrics.items():
        if key not in fair_audit.metrics:
            continue
            
        computed_val = float(fair_audit.metrics[key].value)
        assert computed_val == pytest.approx(target_val, abs=1e-6, nan_ok=True), \
            f"Regression in Raw {key} for {problem_name}/{alg_name}"
