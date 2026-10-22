# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for independently staged, partial historical FAIR diagnostics."""

import numpy as np
import pytest
from calibration.scripts.audit_fair_archive import compute_archive, compute_fair_only


def _gt():
    # Minimization: all points lie on x+y+z=1, with no dominance.
    x = np.linspace(0, 1, 41)
    return np.column_stack((x, 1-x, np.zeros(len(x))))


def test_fair_only_archive_does_not_fabricate_missing_headway_or_quality():
    gt = _gt()
    result = compute_fair_only(gt[::4], gt)
    assert result["k"] == 11
    assert result["metrics"]["CLOSENESS"] == pytest.approx(0.0)
    assert result["metrics"]["HEADWAY"] is None
    assert result["headway_status"] == "unavailable_no_generation_zero"
    assert result["q_scores"] is None
    assert result["q_status"] == "unavailable_no_compatible_calibration"
    for key in ("COVERAGE", "GAP", "REGULARITY", "BALANCE"):
        assert np.isfinite(result["metrics"][key])


def test_fair_only_headway_uses_actual_initial_population():
    gt = _gt()
    final = gt[::4] + np.array([0.02, 0.02, 0.02])
    initial = gt[::4] + np.array([0.1, 0.1, 0.1])
    result = compute_fair_only(final, gt, initial)
    assert 0 < result["metrics"]["HEADWAY"] < 1
    assert result["headway_status"] == "measured_from_gen0"


def test_archived_csv_partial_uses_explicit_raw_gt_and_reports_missing_gen0(tmp_path):
    data, gt_dir = tmp_path/"data", tmp_path/"gt"
    data.mkdir()
    gt_dir.mkdir()
    np.savetxt(gt_dir/"DTLZ1_3_optimal.csv", _gt(), delimiter=",")
    np.savetxt(data/"DTLZ1_NSGA2_standard_run00.csv",
               _gt()[::4], delimiter=",", header="f1,f2,f3", comments="")
    np.savetxt(data/"DTLZ1_NSGA2_standard_run00_gen100.csv",
               _gt()[::4], delimiter=",", header="f1,f2,f3", comments="")
    result = compute_archive(data, gt_dir, study_filter="DTLZ1_NSGA2")
    assert not result["errors"]
    assert len(result["runs"]) == 1
    run = result["runs"][0]
    assert run["metrics"]["CLOSENESS"] == pytest.approx(0.0)
    assert run["metrics"]["HEADWAY"] is None
    assert run["q_scores"] is None
    assert result["cannot_be_used_as_certified_audit_report"] is True
    assert run["gt_source"] == "DTLZ1_3_optimal.csv"

    np.savetxt(data/"DTLZ1_NSGA2_standard_run00_gen0.csv",
               _gt()[::4] + 0.2, delimiter=",", header="f1,f2,f3", comments="")
    updated = compute_archive(data, gt_dir, study_filter="DTLZ1_NSGA2")
    assert updated["runs"][0]["metrics"]["HEADWAY"] is not None


def test_archived_csv_fails_closed_without_matching_gt(tmp_path):
    data, gt_dir = tmp_path/"data", tmp_path/"gt"
    data.mkdir()
    gt_dir.mkdir()
    np.savetxt(data/"DTLZ1_NSGA2_standard_run00.csv",
               _gt()[::4], delimiter=",", header="f1,f2,f3", comments="")
    result = compute_archive(data, gt_dir)
    assert result["runs"] == []
    assert result["errors"][0]["reason"] == "missing matching 3D GT"
