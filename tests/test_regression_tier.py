# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Regression Tier Testing Suite
=======================================

This tier contains strict numerical reproducibility tests against established 
Calibration Reference Data.

Data Source 1 (Points):  `tests/calibration_reference_audit_v0.9.json`
Data Source 2 (Targets): `tests/calibration_reference_targets.json`

Logic: Verifies that the current diagnostic pipeline reproduces the reference
clinical scores and physical metrics for a variety of MOPs and MOEAs.
"""

import pytest
import time
import json
import os
import numpy as np
from MoeaBench.diagnostics import auditor, baselines
from MoeaBench import metrics

# Load Calibration Reference Data (Module Level)
DATA_JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration_reference_audit_v0.9.json")
TARGETS_JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration_reference_targets.json")

print(f"Loading Calibration Reference Data from {DATA_JSON_PATH}...")
with open(DATA_JSON_PATH, "r") as f:
    CALIBRATION_REF_DATA = json.load(f)

print(f"Loading Calibration Reference Targets from {TARGETS_JSON_PATH}...")
with open(TARGETS_JSON_PATH, "r") as f:
    CALIBRATION_REF_TARGETS = json.load(f)

PROBLEMS = sorted(list(CALIBRATION_REF_DATA['problems'].keys()))
ALGORITHMS = ['NSGA2', 'MOEAD', 'NSGA3'] # Standard set in the reference file

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
        assert computed_val == pytest.approx(target_val, abs=1e-6), \
            f"Regression in {key} for {problem_name}/{alg_name}"

    # 4. Verify Raw Fair Metrics
    fair_audit = report.fair_audit_res
    if fair_audit is None:
         pytest.fail(f"Audit failed to produce Fair Metrics for {problem_name}/{alg_name}")

    for key, target_val in target_raw_metrics.items():
        if key not in fair_audit.metrics:
            continue
            
        computed_val = float(fair_audit.metrics[key].value)
        assert computed_val == pytest.approx(target_val, abs=1e-6), \
            f"Regression in Raw {key} for {problem_name}/{alg_name}"
