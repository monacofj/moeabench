# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Positive mathematical controls for independent simplex-space candidates (#119)."""
import numpy as np
import pytest

from moeabench.mops import DTLZ1, DPF1
from scripts.audit.gt_simplex_candidates_issue119 import (
    _objective_image, _sample, candidate, compare, simplex_to_positions,
)


@pytest.mark.parametrize("q",[2,3,5,10])
def test_inverse_simplex_mapping_recovers_the_complete_simplex_and_axes(q):
    rng=np.random.default_rng(12000+q)
    inputs=np.vstack((np.eye(q),rng.dirichlet(np.ones(q),size=64)))
    p=simplex_to_positions(inputs)
    mop=DTLZ1(M=q)
    x=_sample(mop,inputs)
    front=mop.evaluation(x)["F"]
    np.testing.assert_allclose(front,inputs*0.5,rtol=0,atol=3e-15)
    np.testing.assert_allclose(front[:q],np.eye(q)*0.5,rtol=0,atol=3e-15)
    assert np.all((p>=0)&(p<=1))


@pytest.mark.parametrize("name,m,d",[
    ("DTLZ1",3,None),("DTLZ1",5,None),
    ("DPF1",5,2),("DPF1",5,3),("DPF1",10,5),
])
def test_simplex_candidate_is_mathematically_correct_repeatable_and_preserves_extremes(name,m,d):
    mop=DTLZ1(M=m) if name=="DTLZ1" else DPF1(M=m,D=d)
    x,front,error=candidate(mop,64,seed=11923)
    x2,front2,error2=candidate(mop,64,seed=11923)
    np.testing.assert_array_equal(x,x2)
    np.testing.assert_array_equal(front,front2)
    assert error==error2
    assert error<3e-12
    assert front.shape==(64,m)
    assert len(np.unique(front,axis=0))==64
    q=m if name=="DTLZ1" else d
    axes=_objective_image(mop,0.5*np.eye(q))
    np.testing.assert_allclose(front[:q],axes,rtol=0,atol=3e-15)


@pytest.mark.parametrize("name,m,d",[
    ("DTLZ1",3,None),("DTLZ1",5,None),("DPF1",5,3)
])
def test_fixed_budget_objective_space_candidate_study_is_reproducible(name,m,d):
    a=compare(name,m,d,64,probes=256)
    b=compare(name,m,d,64,probes=256)
    for label in ("incumbent","objective_greedy_candidate"):
        for key in ("reference_probe_max","reference_probe_p95",
                    "extreme_max_distance","nearest_neighbor_cv"):
            assert a[label][key]==b[label][key]
            assert a[label][key]>=0
    assert a["objective_greedy_candidate"]["extreme_max_distance"]<3e-15
    assert not a["candidate_proven_continuous_coverage"]


def test_simplex_input_contract_is_explicit():
    with pytest.raises(ValueError):
        simplex_to_positions(np.array([[0.2,0.2]]))
    with pytest.raises(ValueError):
        simplex_to_positions(np.array([[1.,-0.1]]))
