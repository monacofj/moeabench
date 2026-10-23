# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check geometry/distributions without calling built-in MOP Pareto samplers."""
import random
import numpy as np
import pytest
from calibration.scripts.independent_pf_sources_issue75 import (
    sample_independent_pf, surface_identity, normalized_independent_pf,
)


def test_dtlz1_uniform_simplex_is_not_biased_by_original_decision_x_distribution():
    # Uniform Dirichlet(1,1,1) yields expectation f_i = 0.5/3,
    # unlike uniform x1,x2 in the library's decision-space PS sampler.
    F, law=sample_independent_pf("DTLZ1",15000,seed=43)
    assert law=="uniform_simplex_area_dirichlet_111"
    assert surface_identity("DTLZ1",F)["passed"] is True
    np.testing.assert_allclose(np.mean(F,axis=0),np.full(3,1/6),atol=0.009,rtol=0)
    assert np.all(F>0)
    F2,_=sample_independent_pf("DTLZ1",15000,seed=43)
    np.testing.assert_array_equal(F,F2)
    F3,_=sample_independent_pf("DTLZ1",15000,seed=44)
    assert not np.array_equal(F,F3)


def test_dtlz2_sphere_area_sampler_no_polar_angle_concentration():
    F,law=sample_independent_pf("DTLZ2",15000,seed=52)
    assert law=="uniform_positive_sphere_area_z_azimuth"
    assert surface_identity("DTLZ2",F)["passed"] is True
    # Area-uniform octant of the unit sphere has mean 1/2 for each
    # coordinate, unlike incorrectly uniform elevation sampling.
    np.testing.assert_allclose(np.mean(F,axis=0),[.5,.5,.5],atol=.018,rtol=0)
    assert np.all(F>=0)


def test_dpf5_exploratory_decision_sampler_on_true_g0_parset():
    import moeabench as mb
    problem=mb.mops.DPF5(M=3)
    sample,law=sample_independent_pf("DPF5",120,seed=19,mop=problem)
    assert sample.shape==(120,3)
    assert "not_surface_area" in law
    assert surface_identity("DPF5",sample)["passed"] is None
    assert len(np.unique(sample,axis=0))>30


def test_sampler_has_no_builtin_pf_dependency_or_global_rng_mutation():
    # Deliberately no mop passed to DTLZ1/2: no PF methods could be called.
    random.seed(82)
    np.random.seed(83)
    pystate=random.getstate()
    npstate=np.random.get_state()
    for name in ("DTLZ1","DTLZ2"):
        sample_independent_pf(name,50,seed=4)
    assert random.getstate()==pystate
    after=np.random.get_state()
    np.testing.assert_array_equal(after[1],npstate[1])
    assert after[0]==npstate[0] and after[2:]==npstate[2:]


def test_reject_invalid_geometries_and_fail_closed_unknown_mop():
    F,_=sample_independent_pf("DTLZ1",10,seed=0)
    bad=F.copy()
    bad[0] *= 1.1
    with pytest.raises(ValueError,match="surface identity"):
        surface_identity("DTLZ1",bad)
    with pytest.raises(ValueError,match="No externally constructed"):
        sample_independent_pf("DPF3",50,seed=8)
    with pytest.raises(ValueError,match="requires a MOP"):
        sample_independent_pf("DPF5",50,seed=8)
    with pytest.raises(ValueError,match="nonnegative private seed"):
        sample_independent_pf("DTLZ2",50,seed=-1)


def test_independent_gt_freshness_and_preserves_frozen_coordinates():
    raw,_=sample_independent_pf("DTLZ1",200,seed=11)
    gt=raw[:50].copy()
    gt[:,0] += np.linspace(1e-4,1e-3,50)
    normalized,metadata=normalized_independent_pf(
        "DTLZ1",n_points=150,seed=12,
        ideal=np.zeros(3),nadir=np.ones(3),gt=gt,
    )
    assert metadata["surface"]["passed"]
    assert metadata["certified"] is False
    assert metadata["mismatch"]["fresh_count"]==len(normalized)
    assert len(normalized)>=143


def test_dtlz1_builtin_pf_law_is_not_uniform_objective_simplex_area():
    """Mathematically distinct reference measures have distinct f3 moments.

    At the Pareto set, DTLZ1 M=3 yields:
       f1 = .5*x1*x2, f2 = .5*x1*(1-x2), f3 = .5*(1-x1).
    Built-in ps() draws x1,x2 independently Uniform(0,1), hence E[f] =
    (.125,.125,.25). Uniform *simplex area* instead yields (1/6,1/6,1/6).
    This is a distribution-choice issue, not Monte-Carlo rounding.
    """
    import moeabench as mb
    library=np.asarray(mb.mops.DTLZ1(M=3).pf(n_points=20000),dtype=float)
    independent,_=sample_independent_pf("DTLZ1",20000,seed=92)
    assert library.shape==(20000,3)
    np.testing.assert_allclose(np.mean(library,axis=0),[.125,.125,.25],
                               atol=.007,rtol=0)
    np.testing.assert_allclose(np.mean(independent,axis=0),
                               [1/6,1/6,1/6],atol=.007,rtol=0)
    assert np.mean(library[:,2])-np.mean(independent[:,2])>.06
