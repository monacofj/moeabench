# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New nonmutating off-Pareto DPF5 D2 equation diagnostics."""
import pytest
from scripts.audit.gt_dpf5_offpareto_issue119 import audit


@pytest.mark.parametrize("m",[3,4,5,6,10])
def test_offpareto_gamma_scales_spherically(m):
    r=audit(m,64)
    assert r["g_min"]>0
    assert r["D"]["max_abs_objective_difference"] < 1e-10
    assert r["D"]["actual_scaled_norm2_residual"] < 1e-10
    assert not r["global_PF_optimality_certified"]


@pytest.mark.parametrize("m",[3,4])
def test_offpareto_beta_matches_published_d2_equations_for_small_m(m):
    r=audit(m,64)
    assert r["P"]["max_abs_objective_difference"] < 1e-10
    assert r["P"]["actual_scaled_norm2_residual"] < 1e-10


@pytest.mark.parametrize("m",[5,6,10])
def test_offpareto_beta_disagrees_for_higher_m(m):
    r=audit(m,64)
    assert r["P"]["max_abs_objective_difference"] > 1e-5
    assert r["P"]["actual_scaled_norm2_residual"] > 1e-5
    assert r["P"]["reference_scaled_norm2_residual"] < 1e-10
