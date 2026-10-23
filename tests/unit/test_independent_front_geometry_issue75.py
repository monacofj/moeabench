# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #75: independent geometry laws, not repetitions of PF's own RNG."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from calibration.scripts.challenge_independent_front_geometry import (
    independent_physical_front, _normalize_unique_fresh,
    candidate_references,
)


def test_dtlz1_independent_simplex_surface_geometry_and_rng_isolation(monkeypatch):
    # Production DTLZ1.ps() may never be called by this independent oracle.
    from moeabench import mops
    monkeypatch.setattr(mops.DTLZ1,"pf",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("PF sampler used")))
    points = independent_physical_front("DTLZ1", 4096, 59)
    assert points.shape == (4096,3)
    assert np.all(points >= 0)
    np.testing.assert_allclose(points.sum(axis=1), 0.5, atol=1e-14)
    # Permutation symmetry is an independent characteristic of the uniform
    # Dirichlet simplex law; fail spectacularly for an ordered corner-only PF.
    assert np.all(np.abs(points.mean(axis=0) - 1/6) < 0.01)
    np.testing.assert_array_equal(
        points, independent_physical_front("DTLZ1",4096,59))
    assert not np.array_equal(points,independent_physical_front("DTLZ1",4096,60))


def test_dtlz2_independent_positive_sphere_surface_geometry_and_rng_isolation(monkeypatch):
    from moeabench import mops
    monkeypatch.setattr(mops.DTLZ2,"pf",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("PF sampler used")))
    points = independent_physical_front("DTLZ2", 4096, 73)
    assert points.shape == (4096,3)
    assert np.all(points >= 0)
    np.testing.assert_allclose(np.linalg.norm(points,axis=1),1.0,atol=1e-14)
    # Area-uniform unit sphere octant is exchangeable in the coordinates.
    assert np.max(points.mean(axis=0))-np.min(points.mean(axis=0)) < 0.035
    np.testing.assert_array_equal(
        points, independent_physical_front("DTLZ2",4096,73))


def test_dpf5_decision_monte_carlo_does_not_call_internal_ps(monkeypatch):
    from moeabench import mops
    monkeypatch.setattr(mops.DPF5,"ps",
        lambda *a,**k: (_ for _ in ()).throw(AssertionError("PF/PS sampler used")))
    mop = mops.DPF5()
    result = independent_physical_front("DPF5", 1200, 31,mop=mop)
    assert result.shape == (1200,mop.M)
    assert np.all(np.isfinite(result))
    assert not np.array_equal(result,independent_physical_front(
        "DPF5",1200,32,mop=mop))


def test_overlap_duplicated_vectors_and_underfilled_sample_are_rejected():
    x=np.linspace(0,1,101)
    gt=np.column_stack((x,1-x,np.zeros_like(x)))
    good = np.column_stack((np.linspace(0,1,300),
                            1-np.linspace(0,1,300),np.ones(300)*0.01))
    mixed=np.vstack((gt[:5],good,np.repeat(good[:10],4,axis=0)))
    pts=_normalize_unique_fresh(mixed,gt,
          np.zeros(3),np.ones(3),count=200)
    assert pts.shape == (200,3)
    assert len(np.unique(pts,axis=0)) == len(pts)
    assert np.all(cKDTree(gt).query(pts)[0] > 1e-12)
    with pytest.raises(ValueError,match="enough unique"):
        _normalize_unique_fresh(mixed,gt,
               np.zeros(3),np.ones(3),count=301)


def test_cross_generator_requires_independent_fresh_reference():
    # The direct mathematical sampler works for all M=3 analytical
    # constructions, but the *comparison* must reject unsupported MOPs.
    with pytest.raises(ValueError,match="Unsupported MOP"):
        independent_physical_front("DTLZ8",200,12)
