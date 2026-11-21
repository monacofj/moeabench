# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent NEW tests for read-only DPF branch diagnostics (issue #119)."""
import pytest

from scripts.audit.gt_dpf_branch_diagnostics_issue119 import inspect


@pytest.mark.parametrize("m,budget", [(3, 48), (5, 48)])
def test_dpf3_returned_points_match_some_actual_evaluated_ps(m, budget):
    row = inspect("DPF3", m, budget)
    assert row["ps_count"] == budget
    assert row["pf_count"] <= budget
    assert row["pf_subset_of_evaluated_within_1e_10"]
    assert row["returned_pf_internally_dominated"] == 0
    assert not row["globally_pareto_optimal_certified"]
    assert not row["globally_complete_certified"]


@pytest.mark.parametrize("m,budget", [(3, 48), (5, 48)])
def test_dpf5_explicit_branch_accounting_and_finite_replay(m, budget):
    row = inspect("DPF5", m, budget)
    assert row["branches"]["P"]["decision_count"] + row["branches"]["D"]["decision_count"] == row["ps_count"]
    assert row["branches"]["P"]["unique_objective_count"] <= row["branches"]["P"]["decision_count"]
    assert row["branches"]["D"]["unique_objective_count"] <= row["branches"]["D"]["decision_count"]
    assert row["returned_pf_nearest_raw_max"] < 1e-10
    assert row["raw_vs_returned_count_equal"]
    assert not row["globally_complete_certified"]


def test_dpf5_m3_nonsquare_budget_is_recorded_not_normalized_away():
    row = inspect("DPF5", 3, 48)
    assert row["ps_count"] == 36
    assert row["pf_count"] == 36
    assert row["grid_side_if_M3"] == 6


def test_bad_parameters_rejected():
    with pytest.raises(ValueError):
        inspect("DPF5", 2, 48, d=2)
