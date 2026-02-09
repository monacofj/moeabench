# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import numpy as np

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from MoeaBench.diagnostics import auditor
from MoeaBench.diagnostics.enums import DiagnosticStatus


def test_fail_closed_when_baseline_unavailable():
    gt = np.array([
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 1.0],
    ])
    pop = np.array([
        [0.1, 0.9, 0.1],
        [0.6, 0.4, 0.6],
    ])

    original_ref = auditor._load_reference
    original_cal = auditor._load_gt_calibration
    try:
        auditor._load_reference = lambda _: (None, None)
        auditor._load_gt_calibration = lambda _: (None, None)
        res = auditor.audit(pop, ground_truth=gt)
    finally:
        auditor._load_reference = original_ref
        auditor._load_gt_calibration = original_cal

    assert res.status == DiagnosticStatus.UNDEFINED_BASELINE
    assert all(v == "FAIL" for v in res.verdicts.values())


def test_k_grid_never_upsamples():
    assert auditor._snap_to_grid(73) == 50
    raw = np.random.rand(73, 3)
    assert len(auditor._resample_to_K(raw, 100)) == 73


def test_nd_filter_applies_before_metrics():
    gt = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ])
    pop = np.array([
        [0.2, 0.8, 0.2],
        [0.3, 0.9, 0.3],  # dominated by [0.2,0.8,0.2]
        [0.8, 0.2, 0.8],
    ])

    res = auditor.audit(pop, ground_truth=gt)
    assert int(res.metrics.get("n_raw", 0)) == 2
