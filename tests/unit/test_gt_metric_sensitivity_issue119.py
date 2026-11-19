# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent tests for #119 metric sensitivity; historical tests unchanged."""
import numpy as np
import pytest

from scripts.audit.gt_metric_sensitivity_issue119 import (
    compute_metrics, run_case, summarize,
)


@pytest.mark.parametrize("name", ["DTLZ2"])
def test_fixed_real_moea_data_and_gt_sensitive_metrics(name):
    result = run_case(
        name, 3, moea_seed=41, budgets=(12, 24),
        pool_size=60, iterations=2, pop_size=12, generations=3,
    )
    assert len(result["final_population_sha256"]) == 64
    assert result["gt_budgets"] == [12, 24]
    for budget in ("12", "24"):
        assert set(result["metrics"][budget]) == {
            "incumbent", "fps", "fps_mc_cvt"}
        for metrics in result["metrics"][budget].values():
            assert all(np.isfinite(value) for value in metrics.values())
            assert metrics["hv_gt_fixed_bounds"] > 0
    summary = summarize([result], largest_budget=24)
    assert summary[0]["moea_seeds"] == 1
    assert summary[0]["methods"]["fps"]["24"]["igd"]["median_abs_change_vs_largest"] == 0


def test_hv_sol_is_invariant_under_only_gt_changes():
    # Raw HV of a fixed population in fixed bounds cannot depend on GT.
    front = np.array([[0.3, 0.8], [0.8, 0.3]])
    gt_a = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    gt_b = np.array([[0.0, 1.0], [0.25, 0.75], [0.5, 0.5],
                     [0.75, 0.25], [1.0, 0.0]])
    x = compute_metrics(front, gt_a, target=1.0)
    y = compute_metrics(front, gt_b, target=1.0)
    assert x["hv_solution_fixed_bounds"] == y["hv_solution_fixed_bounds"]
    assert x["hv_gt_fixed_bounds"] != y["hv_gt_fixed_bounds"]
    assert x["hv_absolute_ratio_fixed_bounds"] != y["hv_absolute_ratio_fixed_bounds"]
