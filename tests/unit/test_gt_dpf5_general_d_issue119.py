# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent new DPF5 M>D>=2 published-formula tests, no regression edits."""
import numpy as np
import pytest
from scripts.audit.gt_dpf5_general_d_issue119 import spherical, reference, check


@pytest.mark.parametrize("d,m",[(2,3),(2,5),(3,4),(3,5),(3,6),(3,8),(4,5),(4,6),(4,8)])
def test_independent_formula_satisfies_scaled_sphere_for_general_d(d,m):
    r=check(m,d,64,0.2)
    assert r["branch_P"]["independent_scaled_sphere_residual"] < 1e-12
    assert r["branch_D"]["independent_scaled_sphere_residual"] < 1e-12
    assert not r["global_pareto_completeness_certified"]


@pytest.mark.parametrize("m,d",[(3,2),(4,3)])
def test_nearest_dimension_beta_and_gamma_agree_with_current_interior(m,d):
    r=check(m,d,64,0.0)
    assert r["branch_P"]["max_objective_difference"] < 1e-12
    assert r["branch_D"]["max_objective_difference"] < 1e-12


def test_general_d_gamma_is_replicated_first_essential_coordinate():
    x=np.full((3,9),0.5)
    x[:,0]=[0.6,0.7,0.8]
    x[:,5]=x[:,0]
    f=reference(x,6,3)
    np.testing.assert_allclose(f[:,0],f[:,1],atol=1e-12)
    np.testing.assert_allclose(f[:,1],f[:,2],atol=1e-12)
    np.testing.assert_allclose(f[:,2],f[:,3],atol=1e-12)
    np.testing.assert_allclose(np.sum(f*f,axis=1),1,atol=1e-12)


def test_high_m_general_d_current_p_diverges():
    r=check(8,3,64,0.0)
    assert r["branch_P"]["max_objective_difference"] > 1e-5
    assert r["branch_P"]["actual_scaled_sphere_residual"] > 1e-5

def test_d4_gamma_source_disagrees_with_published_equations():
    r=check(6,4,64,0.0)
    assert r["intrinsic_D"] == 4
    assert r["branch_D"]["max_objective_difference"] > 1e-5
    assert r["branch_D"]["actual_scaled_sphere_residual"] > 1e-5
