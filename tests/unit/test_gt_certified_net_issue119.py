# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent research tests for issue #119 analytic covering bounds."""
from math import comb
import numpy as np
import pytest
from scipy.spatial import cKDTree

from scripts.audit.gt_certified_net_issue119 import (
    analytical_epsilon, barycentric_lattice, coverage_interval, audit,
)
from scripts.audit.gt_exact_curve_issue119 import exact_curve_stats


@pytest.mark.parametrize("m,q",[(2,4),(3,8),(5,5)])
def test_lattice_enumerates_all_compositions_and_axes(m,q):
    x=barycentric_lattice(m,q)
    assert x.shape==(comb(q+m-1,m-1),m)
    np.testing.assert_allclose(x.sum(axis=1),1,rtol=0,atol=1e-15)
    assert len(np.unique(x,axis=0))==len(x)
    for axis in np.eye(m):
        assert np.any(np.all(np.isclose(x,axis),axis=1))


@pytest.mark.parametrize("m,q",[(3,4),(5,5)])
def test_epsilon_net_bound_holds_on_independent_random_surface_points(m,q):
    rng=np.random.default_rng(177)
    u=rng.dirichlet(np.ones(m),size=300)
    lattice=barycentric_lattice(m,q)
    for name in ("DTLZ1","DTLZ2"):
        if name=="DTLZ1":
            ref=u*0.5
            net=lattice*0.5
        else:
            ref=u/np.linalg.norm(u,axis=1,keepdims=True)
            net=lattice/np.linalg.norm(lattice,axis=1,keepdims=True)
        observed=cKDTree(net).query(ref,k=1)[0]
        assert max(observed)<=analytical_epsilon(name,m,q)+1e-12


@pytest.mark.parametrize("name,points",[
    ("DTLZ1", np.array([[0.5,0.0],[0.0,0.5]])),
    ("DTLZ2", np.array([[1.0,0.0],[0.0,1.0]])),
])
def test_new_bounds_enclose_independent_exact_continuous_m2_oracle(name,points):
    true=exact_curve_stats(points,name)["exact_continuous_covering_radius"]
    for q in (4,8,16):
        x=coverage_interval(points,name,q=q)
        assert x["observed_net_max_lower"] <= true+1e-12
        assert x["continuous_radius_upper"] >= true-1e-12
        assert x["interval_width"]<=analytical_epsilon(name,2,q)+1e-12+1e-15


@pytest.mark.parametrize("name",["DTLZ1","DTLZ2"])
def test_real_mop_fps_and_incumbent_bounds_are_valid(name):
    report=audit(name,3,16,q_values=(4,8),candidate_pool=48,seed=3,max_points=100)
    for method in ("incumbent","fps"):
        x=report["methods"][method]
        for q in ("4","8"):
            assert x[q]["GT_points"]==16
            assert x[q]["observed_net_max_lower"]<=x[q]["continuous_radius_upper"]
            assert x[q]["mean_distance_certified"] is False


def test_large_lattice_guard_and_invalid_off_surface_gt():
    with pytest.raises(ValueError,match="Refusing lattice"):
        barycentric_lattice(5,24,max_points=500)
    with pytest.raises(ValueError,match="off analytical simplex"):
        coverage_interval(np.array([[0.2,0.2,0.2]]),"DTLZ1",q=4)
    with pytest.raises(ValueError,match="off positive unit sphere"):
        coverage_interval(np.array([[0.5,0.5,0.5]]),"DTLZ2",q=4)
