# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent tests of dependent-rounding GT net proof (#119)."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from scripts.audit.gt_certified_net_issue119 import (
    analytical_epsilon as prior_bound, barycentric_lattice,
)
from scripts.audit.gt_exact_curve_issue119 import exact_curve_stats
from scripts.audit.gt_refined_net_issue119 import (
    dependent_rounding_epsilon, improved_coverage_interval, comparison,
)


@pytest.mark.parametrize("m,q",[(2,3),(3,8),(5,7),(8,4)])
def test_constructive_systematic_rounding_maintains_k_and_average(m,q):
    """Verify the proof's marginals and fixed sum for an independent sample."""
    rng=np.random.default_rng(717)
    for _ in range(15):
        u=rng.dirichlet(np.ones(m))
        a=q*u
        floors=np.floor(a).astype(int)
        fractional=a-floors
        k=q-int(floors.sum())
        # Exact enumeration of a finite set of offset intervals
        boundaries=np.unique(np.r_[0.0,1.0,np.mod(-np.cumsum(fractional),1)])
        middles=(boundaries[:-1]+boundaries[1:])/2
        weights=np.diff(boundaries)
        rounded=[]
        for t in middles:
            sums=np.r_[0.0,np.cumsum(fractional)]
            bits=np.floor(sums[1:]+t)-np.floor(sums[:-1]+t)
            rounded.append(bits)
        bits=np.asarray(rounded)
        np.testing.assert_allclose(bits.sum(axis=1),k,atol=1e-10)
        np.testing.assert_allclose(weights@bits,fractional,atol=1e-10)
        errors=np.sum((bits-fractional)**2,axis=1)
        assert np.min(errors) <= m/4 + 1e-10


@pytest.mark.parametrize("m,q",[(3,4),(5,7)])
def test_refined_epsilon_covers_independent_simplex_and_sphere_points(m,q):
    grid=barycentric_lattice(m,q)
    rng=np.random.default_rng(24680)
    u=rng.dirichlet(np.ones(m),size=300)
    for name in ("DTLZ1","DTLZ2"):
        if name=="DTLZ1":
            net=grid/2; x=u/2
        else:
            net=grid/np.linalg.norm(grid,axis=1,keepdims=True)
            x=u/np.linalg.norm(u,axis=1,keepdims=True)
        assert max(cKDTree(net).query(x,k=1)[0]) <= dependent_rounding_epsilon(name,m,q)+1e-12
        np.testing.assert_allclose(
            2*dependent_rounding_epsilon(name,m,q),
            prior_bound(name,m,q),rtol=0,atol=1e-13)


@pytest.mark.parametrize("name,points",[
    ("DTLZ1",np.array([[.5,0.],[0.,.5]])),
    ("DTLZ2",np.array([[1.,0.],[0.,1.]])),
])
def test_new_bounds_enclose_independent_exact_curve_truth(name,points):
    truth=exact_curve_stats(points,name)["exact_continuous_covering_radius"]
    for q in (4,8,16):
        new=improved_coverage_interval(points,name,q=q)
        assert new["observed_net_max_lower"]<=truth+1e-12
        assert new["continuous_radius_upper"]>=truth-1e-12
        assert new["interval_width"] < new["previous_interval_width"]
        assert new["mean_distance_certified"] is False


@pytest.mark.parametrize("name",["DTLZ1","DTLZ2"])
def test_actual_mop_candidate_and_original_bounds_reproducible(name):
    x=comparison(name,3,16,q_values=(4,8),candidate_pool=48,seed=101,max_points=100)
    y=comparison(name,3,16,q_values=(4,8),candidate_pool=48,seed=101,max_points=100)
    assert x==y
    for method in ("incumbent","fps"):
        for q in ("4","8"):
            s=x["methods"][method][q]
            assert s["net_points"] > 0
            assert s["continuous_radius_upper"] >= s["observed_net_max_lower"]


def test_bad_parameters_are_rejected():
    with pytest.raises(ValueError):
        dependent_rounding_epsilon("OTHER",3,8)
    with pytest.raises(ValueError):
        improved_coverage_interval(np.ones((3,3)),"DTLZ1",q=8)
