# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent GT cardinality, reproducibility, Pareto membership and coverage.

These are NEW tests for issue #119's corrected MOP-specific finite samplers.
Historical regression tests and reference baseline JSON remain untouched.
"""

import numpy as np
import pytest

from moeabench.mops.DPF3 import DPF3
from moeabench.mops.DPF5 import DPF5


@pytest.mark.parametrize("n",[2,31,100,128,512])
def test_dpf3_d2_angular_gt_has_requested_unique_objectives(n):
    mop=DPF3(M=5,D=2)
    ps=mop.ps(n_points=n)
    pf=mop.pf(n_points=n)
    assert ps.shape==(n,mop.N)
    assert pf.shape==(n,mop.M)
    assert len(np.unique(pf,axis=0))==n
    assert np.isfinite(pf).all()
    assert np.array_equal(ps,mop.ps(n_points=n))
    assert np.array_equal(pf,mop.pf(n_points=n))
    np.testing.assert_array_equal(ps[:,1:],np.full((n,mop.N-1),.5))


@pytest.mark.parametrize("m,d,n",[(5,3,128),(6,4,128),(7,3,256)])
def test_dpf3_higher_d_gt_remains_globally_optimal_and_unique(m,d,n):
    mop=DPF3(M=m,D=d)
    ps=mop.ps(n_points=n)
    pf=mop.evaluation(ps)["F"]
    assert ps.shape==(n,mop.N)
    assert pf.shape==(n,m)
    assert len(np.unique(pf,axis=0))==n
    assert np.isfinite(pf).all()
    assert np.all(ps>=0) and np.all(ps<=1)
    np.testing.assert_array_equal(ps[:,d-1:],np.full((n,mop.N-d+1),.5))
    np.testing.assert_array_equal(ps,mop.ps(n_points=n))


def test_dpf3_d2_angular_coverage_under_exact_arithmetic_bound_with_roundoff():
    mop=DPF3(M=3,D=2)
    n=128
    reference=mop.pf(n_points=n)
    probes=np.linspace(0,np.pi/2,2001)
    x=np.full((len(probes),mop.N),.5)
    x[:,0]=np.power(probes/(np.pi/2),1/100)
    objectives=mop.evaluation(x)["F"]
    from scipy.spatial import cKDTree
    maximum=np.max(cKDTree(reference).query(objectives,k=1)[0])
    assert maximum<=np.pi/(4*(n-1))+1e-11


@pytest.mark.parametrize("m,d,n",[(3,2,128),(3,2,512),(5,2,128),
                                     (6,3,128),(6,4,128),(7,3,256)])
def test_dpf5_branchwise_ps_has_unique_objectives_and_both_components(m,d,n):
    mop=DPF5(M=m,D=d)
    ps=mop.ps(n_points=n)
    pf=mop.evaluation(ps)["F"]
    assert ps.shape==(n,mop.N)
    assert pf.shape==(n,m)
    assert np.isfinite(pf).all()
    assert len(np.unique(pf,axis=0))==n
    assert np.any(ps[:,0]<1/3)
    assert np.any(ps[:,0]>=1/3)
    assert np.array_equal(ps,mop.ps(n_points=n))
    np.testing.assert_allclose(np.linalg.norm(pf,axis=1),1,atol=5e-14,rtol=0)
    np.testing.assert_array_equal(ps[:,m-1],ps[:,0])
    if mop.N>m:
        np.testing.assert_array_equal(ps[:,m:],np.full((n,mop.N-m),.5))


def test_dpf5_boundary_is_present_on_degenerate_branch():
    mop=DPF5(M=5,D=2)
    ps=mop.ps(n_points=128)
    assert np.any(ps[:,0]==1/3)
    assert np.any(ps[:,0]<1/3)
    pf=mop.evaluation(ps)["F"]
    assert np.isfinite(pf).all()


@pytest.mark.parametrize("mop_type", [DPF3, DPF5])
def test_gt_samplers_reject_invalid_budgets(mop_type):
    mop=mop_type()
    with pytest.raises(ValueError):
        mop.ps(n_points=0)
    with pytest.raises(ValueError):
        mop.ps(n_points=-1)
    with pytest.raises(ValueError):
        mop.ps(n_points=3.5)
