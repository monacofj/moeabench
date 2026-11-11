# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #78: analytical DTLZ7/DPF2 Pareto source, not a baseline migration."""
import itertools

import numpy as np
import pytest
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from moeabench.mops import DPF2, DTLZ7
from moeabench.mops._dtlz7_pareto_regions import (
    _benefit, dtlz7_record_intervals, map_uniform_to_record_intervals,
    sample_dtlz7_record_parameters,
)


def _nd_fraction(values):
    values = np.asarray(values, dtype=float)
    unique = np.unique(values, axis=0)
    idx = NonDominatedSorting().do(unique,only_non_dominated_front=True)
    return len(idx)/len(unique)


def _mask_components(parameters):
    (l0,h0),(l1,h1)=dtlz7_record_intervals()
    assert np.all((parameters>=l0-1e-12)&(parameters<=h1+1e-12))
    assert not np.any((parameters>h0+1e-12)&(parameters<l1-1e-12))
    return (parameters>l1).astype(int)


def test_analytical_intervals_match_independent_published_endpoints():
    (lo1,hi1),(lo2,hi2)=dtlz7_record_intervals()
    assert lo1==0
    assert hi1==pytest.approx(0.2514118360889171,abs=1e-10)
    assert lo2==pytest.approx(0.6316265307000614,abs=1e-10)
    assert hi2==pytest.approx(0.8594008566447239,abs=1e-10)
    assert _benefit(hi1)==pytest.approx(_benefit(lo2),abs=1e-11)
    # At the record reentry point itself, x=lo2 is weakly dominated by hi1.
    assert _benefit(hi2)>_benefit(hi1)


def test_mapping_rejects_illegal_inputs_and_does_not_use_global_rng():
    for values in ([-.001],[1.001],[np.nan]):
        with pytest.raises(ValueError,match="uniform variates"):
            map_uniform_to_record_intervals(values)
    s=np.random.get_state()
    a=sample_dtlz7_record_parameters(1024,2,seed=42)
    b=sample_dtlz7_record_parameters(1024,2,seed=42)
    assert np.array_equal(a,b)
    t=np.random.get_state()
    assert all(np.array_equal(i,j) if isinstance(i,np.ndarray) else i==j
               for i,j in zip(s,t))
    _mask_components(a)
    with pytest.raises(ValueError,match="Structured"):
        sample_dtlz7_record_parameters(32,2,structured=True)


@pytest.mark.parametrize("M", [2,3,4])
def test_dtlz7_true_pareto_regions_and_every_component_represented(M):
    mop=DTLZ7(M=M)
    n=2048
    x=mop.ps(n)
    assert x.shape==(n,mop.N)
    assert np.all(x[:,M-1:]==0.0)
    masks=_mask_components(x[:,:M-1])
    components={tuple(z) for z in masks}
    assert components==set(itertools.product((0,1),repeat=M-1))
    front=mop.pf(n)
    assert front.shape==(n,M)
    assert _nd_fraction(front)==1.0
    np.testing.assert_allclose(mop.evaluation(x)["F"],front,atol=0,rtol=0)


@pytest.mark.parametrize("D,M", [(2,3),(3,4)])
def test_dpf2_true_pareto_regions_preserve_monotone_projection(D,M):
    mop=DPF2(D=D,M=M)
    n=1024
    x=mop.ps(n)
    assert x.shape==(n,mop.N)
    assert np.all(x[:,D-1:]==0.0)
    masks=_mask_components(x[:,:D-1])
    assert {tuple(z) for z in masks}==set(itertools.product((0,1),repeat=D-1))
    front=mop.pf(n)
    assert front.shape==(n,M)
    assert _nd_fraction(front)==1.0
    np.testing.assert_allclose(front,mop.evaluation(x)["F"],atol=0,rtol=0)
    if D==2:
        assert np.all(np.diff(x[:,0])>=0)


def test_old_unrestricted_generators_were_demonstrably_dominated():
    d=DTLZ7(M=3)
    raw=np.zeros((1024,d.N))
    raw[:,:2]=np.random.RandomState(42).random((1024,2))
    assert _nd_fraction(d.evaluation(raw)["F"])<.5
    p=DPF2(D=2,M=3)
    raw2=np.zeros((1024,p.N))
    raw2[:,0]=np.linspace(0.,1.,1024)
    assert _nd_fraction(p.evaluation(raw2)["F"])<.6


def test_record_property_global_dominance_against_all_smaller_parameters():
    # The *reason* each sampled point is truly Pareto-optimal, beyond the
    # finite within-sample ND test, is its strict prefix-record property.
    xs=np.linspace(0.0,1.0,100001)
    sample=sample_dtlz7_record_parameters(512,1).ravel()
    for x in sample:
        preceding=xs[xs<x-1e-8]
        if len(preceding):
            assert _benefit(x)>np.max(_benefit(preceding))-1e-7
