# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue 75: calibrate only quantities definable from one population."""

import numpy as np
import pytest

from calibration.scripts import benchmark_fair_calibration_cost as cost


def test_physical_one_population_probe_omits_headway():
    # Non-dominated 3-objective GT: all points lie on f1+f2=1.
    x = np.linspace(0, 1, 40)
    gt = np.column_stack((x, 1-x, np.zeros(len(x))))
    p = gt[::4]
    u_ref = gt[::4]
    centres = u_ref.copy()
    ref_hist = np.ones(len(centres))/len(centres)
    result = cost._measure(p, gt, u_ref, centres, ref_hist)
    assert set(result) == set(cost.METRICS)
    assert "HEADWAY" not in result
    assert all(np.isfinite(value) for value in result.values())
    assert result["COVERAGE"] >= 0
    assert result["GAP"] >= 0


def test_cost_summary_contains_no_pseudo_headway(monkeypatch):
    def fake_problem(name, **kwargs):
        if name == "invalid":
            raise ValueError("reference mismatch")
        return {"problem": name, "headway": None,
                "headway_status": "NOT_CALIBRATED_requires_true_P0_Pfinal_pairs"}
    monkeypatch.setattr(cost, "benchmark_problem", fake_problem)
    report = cost.run_benchmark(["DTLZ1", "invalid"], k=20)
    assert report["cannot_replace_official_baselines"] is True
    assert report["headway_omitted"] is True
    assert len(report["problem_results"]) == 1
    assert "invalid" in report["failures"]


def test_invalid_sampling_inputs_fail_before_expensive_calibration():
    with pytest.raises(ValueError, match="sampling"):
        cost.benchmark_problem("DTLZ1", k=1)
