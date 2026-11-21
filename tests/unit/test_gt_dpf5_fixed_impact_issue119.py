# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New research-only tests for DPF5 paired isolation; no historical edits."""
import pytest
from scripts.audit.gt_dpf5_fixed_impact_issue119 import audit


@pytest.mark.parametrize("m,d",[(3,2),(5,2),(6,3),(6,4)])
def test_fixed_x_and_fixed_population_isolation(m,d):
    row=audit(m,d,48,12,101)
    assert len(row["fixed_population_sha256"]) == 64
    assert row["current_reference"]["h_sol"] == row["published_formula_counterfactual"]["h_sol"]
    assert row["gt_pareto_membership_certified"] is False
    assert row["gt_pareto_completeness_certified"] is False


@pytest.mark.parametrize("m,d",[(5,2),(6,3),(6,4)])
def test_corrected_source_agrees_with_published_model_at_fixed_decisions(m,d):
    row=audit(m,d,48,12,101)
    assert row["gt_max_component_change"] < 5e-13
    assert row["population_max_component_change_at_fixed_X"] < 5e-13
    assert abs(row["metric_delta_counterfactual_minus_current"]["igd"]) < 1e-11
    assert abs(row["metric_delta_counterfactual_minus_current"]["igdplus"]) < 1e-11


def test_m3_d2_interior_control_agrees():
    row=audit(3,2,48,12,101)
    assert row["gt_max_component_change"] < 1e-11
    assert row["population_max_component_change_at_fixed_X"] < 1e-11
