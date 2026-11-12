# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #83: GT half-normal blur must be one-sided in Pareto minimization."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from calibration.scripts.generate_baselines import (
    dominance_outward_direction,
    estimate_gt_normals,
    generate_half_normal_blur,
    calibrate_sigma,
)
from moeabench.mops import DTLZ2, DTLZ7, DPF5
from moeabench.diagnostics.baselines import get_resolution_factor_k


def _same_rng_state(before, after):
    return all(np.array_equal(x,y) if isinstance(x,np.ndarray) else x==y
               for x,y in zip(before,after))


def test_minimization_dominance_oriented_outward_projection():
    for n in ([1,2,3],[-1,-2,-3],[1,-2,0],[-1,2,0],[0,0,0]):
        d=dominance_outward_direction(n)
        assert np.all(d>=0.0)
        assert np.isclose(np.linalg.norm(d),1.0,atol=1e-12)
    # Sign must be chosen independent of the (possibly nonconvex) GT point.
    np.testing.assert_allclose(
        dominance_outward_direction(np.array([-2.,-2.])),
        np.array([1.,1.])/np.sqrt(2.0))
    np.testing.assert_allclose(
        dominance_outward_direction(np.array([1.,-2.])),
        np.array([0.,1.]))
    np.testing.assert_allclose(
        dominance_outward_direction(np.array([-2.,1.])),
        np.array([1.,0.]))
    np.testing.assert_allclose(
        dominance_outward_direction(np.array([0.,0.])),
        np.array([1.,1.])/np.sqrt(2.0))


@pytest.mark.parametrize("invalid",[[np.nan,1],[np.inf,0],[],[[1.,2.]]])
def test_invalid_normal_refused(invalid):
    with pytest.raises(ValueError,match="finite, nonempty"):
        dominance_outward_direction(invalid)


def test_one_sided_half_normal_never_improves_any_objective():
    # This oracle fails against the legacy sign=random(-1,+1) generator:
    # a negative draw could be accepted if coordinates stayed >=0.
    gt=np.ones((128,3),dtype=float)*2
    directions=np.tile([1.,1.,1.],(128,1))
    a=generate_half_normal_blur(gt,directions,.4,np.random.RandomState(11))
    b=generate_half_normal_blur(gt,directions,.4,np.random.RandomState(11))
    np.testing.assert_array_equal(a,b)
    assert np.all(a>=gt)
    assert np.all(np.any(a>gt,axis=1))
    assert np.all(np.isfinite(a))
    assert np.all(a>=0)


def test_private_seed_does_not_mutate_numpy_global_random_state():
    gt=np.ones((32,3),dtype=float)
    n=np.tile([1.,-2.,0.],(32,1))
    before=np.random.get_state()
    a=generate_half_normal_blur(gt,n,.3,np.random.RandomState(4))
    after=np.random.get_state()
    assert _same_rng_state(before,after)
    assert np.all(a>=gt)
    np.testing.assert_array_equal(a[:,0],gt[:,0])
    np.testing.assert_array_equal(a[:,2],gt[:,2])


def test_bad_sigma_or_invalid_gt_refused():
    gt=np.ones((4,3))
    n=np.ones((4,3))
    for sigma in [0.,-1.,np.nan,np.inf]:
        with pytest.raises(ValueError,match="sigma"):
            generate_half_normal_blur(gt,n,sigma,np.random.RandomState(1))
    with pytest.raises(ValueError,match="matching finite"):
        generate_half_normal_blur(gt,n[:3],.1,np.random.RandomState(1))
    with pytest.raises(ValueError,match="matching finite"):
        generate_half_normal_blur(-gt,n,.1,np.random.RandomState(1))


@pytest.mark.parametrize("mop",[
    DTLZ2(M=3),
    DTLZ7(M=3),
    DPF5(D=2,M=3),
])
def test_real_analytical_pf_outward_no_improvement(mop):
    gt=np.asarray(mop.pf(n_points=512),dtype=float)
    # Actual source PF objective vectors are all nonnegative.
    assert np.min(gt)>=-1e-14
    normals=estimate_gt_normals(gt)
    assert normals.shape==gt.shape
    assert np.all(np.isfinite(normals))
    assert np.all(normals>=0)
    np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1.,atol=1e-12)
    blurred=generate_half_normal_blur(gt,normals,.15,np.random.RandomState(42))
    assert blurred.shape==gt.shape
    assert np.all(blurred>=gt)
    assert np.all(np.any(blurred>gt,axis=1))


def test_half_normal_sigma_targets_two_resolution_units():
    gt=np.asarray(DTLZ2(M=3).pf(n_points=512),dtype=float)
    normals=estimate_gt_normals(gt)
    sk=get_resolution_factor_k(gt,100,seed=0)
    assert sk>0
    sigma,calibration_median,p95=calibrate_sigma(
        gt,normals,sk,rng_seed=1200)
    assert sigma>0 and np.isfinite(sigma)
    assert p95>=calibration_median
    # Existing 6-iteration controller aims for 2 with 5% convergence.
    # Bound the calibration rule rather than demanding an exact draw.
    assert abs(calibration_median-2.)<.35
    actual=generate_half_normal_blur(
        gt,normals,sigma,np.random.RandomState(1700))
    observed=np.median(cKDTree(gt).query(actual,k=1)[0]/sk)
    assert 1.5<observed<2.5
    assert np.all(actual>=gt)


def test_pca_on_small_GT_does_not_query_nonexistent_neighbors():
    gt=np.array([[1.,0.],[0.,1.],[.5,.5]])
    n=estimate_gt_normals(gt)
    assert n.shape==gt.shape
    assert np.all(n>=0)
    with pytest.raises(ValueError,match="at least two"):
        estimate_gt_normals(gt[:1])
