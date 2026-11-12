# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Representation oracles and evidence-only integration for issue #81."""
from itertools import combinations_with_replacement
import numpy as np
import pytest
from scipy.stats import wasserstein_distance
from calibration.scripts.study_dpf5_ecdf_compression_issue81 import (
    METHODS, represent, numerical_sign, pairwise, errors, variance_identity, study,
)


@pytest.mark.parametrize('method', METHODS)
def test_full_resolution_is_exact_with_multiplicity_and_zero_mass(method):
    values=np.array([3.,0.,1.,0.,8.,3.])
    np.testing.assert_array_equal(represent(values,len(values),method),np.sort(values))


def test_compatibility_and_linear_endpoint_definitions_differ_as_expected():
    z=np.arange(6.)
    np.testing.assert_array_equal(represent(z,4,'compatibility_floor'),[0,1,3,5])
    np.testing.assert_allclose(represent(z,4,'endpoint_linear'),[0,5/3,10/3,5])


def test_quantile_midpoints_minimize_w1_over_equal_weight_empirical_support():
    z=np.array([0.,0.,1.,3.,20.])
    mids=represent(z,2,'midpoint_inverse')
    np.testing.assert_array_equal(mids,[0,3])
    optimum=min(wasserstein_distance(z,c) for c in combinations_with_replacement(z,2))
    assert wasserstein_distance(z,mids)==optimum


def test_bin_means_integrate_nondivisible_probability_cells():
    z=np.array([0.,3.,12.])
    np.testing.assert_array_equal(represent(z,2,'bin_mean'),[1.,9.])
    assert represent(z,2,'bin_mean').mean()==z.mean()


@pytest.mark.parametrize('method',METHODS)
def test_reduced_representation_is_ordered_and_bounded(method):
    z=np.array([0,0,.1,.5,1,3,3,10.])
    result=represent(z,3,method)
    assert len(result)==3 and np.all(np.diff(result)>=0)
    assert result.min()>=z.min() and result.max()<=z.max()


@pytest.mark.parametrize('values,m,method',[
    ([0.,1.],0,'bin_mean'),([0.,1.],3,'bin_mean'),
    ([0.,float('nan')],2,'bin_mean'),([-1.,1.],2,'bin_mean'),
    ([0.,1.],2,'unknown'),
])
def test_invalid_laws_rejected(values,m,method):
    with pytest.raises(ValueError):represent(values,m,method)


def test_float_resolution_ties_are_distinct_from_strict_reversals():
    a=.5
    assert numerical_sign(a,a)==0
    assert numerical_sign(a,np.nextafter(a,1.))==0
    assert numerical_sign(a,.500001)==-1
    full=np.array([[.51,.51],[.50,.50]])
    q=np.array([[.49,.51],[.50,.50]])
    p=pairwise(q,full)
    assert p['strict_reversing_pairs']==1
    assert p['disagreeing_pair_calibration_cells']==1
    assert p['numerical_tie_nonexact_cells']==0


def test_relative_error_has_explicit_zero_denominator_handling():
    v=errors(np.array([.1,.5]),np.array([0.,.5]))
    assert v['relative_undefined_full_zero']==1
    assert v['relative_absolute']['n']==1
    assert v['relative_absolute']['max']==0


def test_variance_identity_includes_covariance():
    full=np.array([[.1,.3,.5],[.6,.7,.8]])
    q=full+np.array([[.1,0,-.1],[-.1,0,.1]])
    v=variance_identity(q,full)
    assert v['covariance'][0]<0 and v['covariance'][1]>0
    assert v['identity_max_residual']<1e-16


def test_saved_evidence_study_never_optimizes_refits_or_resamples_ideal(monkeypatch):
    import moeabench as mb
    import calibration.scripts.generate_baselines as generator
    import calibration.scripts.validate_dpf5_closeness_protocol_issue81 as validator
    def forbidden(*args,**kwargs):
        raise AssertionError('New optimization/calibration is forbidden')
    monkeypatch.setattr(mb,'experiment',forbidden)
    monkeypatch.setattr(generator,'calibrate_sigma',forbidden)
    monkeypatch.setattr(validator,'sample_independent_ideal',forbidden)
    report,rows,pairs,refs=study()
    assert report['verification']['replayed_original_blurs']==24
    assert report['verification']['exact_previous_full_q_cells']==384
    assert len(rows)==4*5*384
    assert report['verification']['max_scipy_q_discrepancy']<1e-12
    for k,cell in report['results_by_K'].items():
        assert refs[k].shape==(8,3367)
        assert cell['full_pairwise_summary']['strict_reversing_pairs']==0
        for method in METHODS:
            assert cell['methods'][method]['3367']['errors']['absolute']['max']==0
            assert cell['methods'][method]['3367']['ranking']['disagreeing_pairs']==0


def test_bin_mean_w1_contraction_requires_population_quantile_alignment():
    from calibration.scripts.study_dpf5_real_q_full_calibrations_issue81 import q_from_raw
    # m=4 is a multiple of population size 2: its quantile is constant on
    # each compression bin, so Jensen gives d_r(compressed) <= d_r(full).
    law=np.array([0.,.1,.8,1.,3.,8.,9.])
    population=np.array([.1,2.])
    reduced=represent(law,4,'bin_mean')
    assert wasserstein_distance(population,reduced)<=wasserstein_distance(population,law)
    assert q_from_raw(population,0,reduced)<=q_from_raw(population,0,law)
    # Nonaligned bins need not contract W1 to the optimizer population.
    law=np.array([0.,1.,2.,3.])
    population=np.array([0.,.9,1.8,2.7])
    reduced=represent(law,2,'bin_mean')
    assert q_from_raw(population,0,reduced)==.25
    assert q_from_raw(population,0,law)==pytest.approx(.1)
