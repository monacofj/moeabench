# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent research assertions: no historical regression edited."""
import numpy as np
import pytest

from scripts.audit.gt_dpf_numerics_kernel_issue119 import (
    dpf3_precision, dpf5_kernel, canonical_dpf5_pf_from_ps,
)


def test_dpf3_corrected_source_preserves_small_positive_complement():
    from moeabench.mops.DPF3 import DPF3
    mop = DPF3(M=3,D=2)
    x = np.full((1,mop.N),0.5)
    x[0,0] = 0.7
    theta=(np.pi/2)*(0.7**100)
    assert 1 - np.cos(theta) == 0
    expected=2*np.sin(theta/2)**2
    assert expected>0
    np.testing.assert_allclose(mop.evaluation(x)["F"][0,0],expected,rtol=1e-12,atol=0)
    row=dpf3_precision(128,3)
    assert row["stable_zero_count"] <= row["direct_zero_count"]
    assert row["raw_nondominated"] <= 128


@pytest.mark.parametrize("m", [3,5])
def test_published_sphere_identity_holds_for_counterfactual(m):
    # Valid g=0 PS with an interior P branch; all beta coordinates active.
    ps=np.full((3,max(m+3,7)),0.5)
    ps[:,0]=[0.1,0.2,0.3]
    ps[:,m-1]=ps[:,0]
    f=canonical_dpf5_pf_from_ps(ps,m)
    np.testing.assert_allclose(np.sum(f**2,axis=1),1,rtol=0,atol=1e-12)


def test_dpf5_m3_agrees_away_from_boundary():
    row=dpf5_kernel(128,3)
    assert row["branch_P"]["actual_vs_canonical_max_abs_component"] < 1e-12
    assert row["branch_D"]["actual_vs_canonical_max_abs_component"] < 1e-12


def test_dpf5_m5_beta_formula_matches_published_sphere():
    row=dpf5_kernel(128,5)
    assert row["branch_P"]["actual_vs_canonical_max_abs_component"] < 1e-12
    assert row["branch_P"]["actual_max_unit_sphere_residual"] < 1e-12
    assert row["branch_P"]["canonical_max_unit_sphere_residual"] < 1e-10
    assert row["branch_D"]["actual_vs_canonical_max_abs_component"] < 1e-12
    assert not row["global_pareto_completeness_certified"]
