# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Positive component-presence guarantees for candidate, not production GT."""
import numpy as np
import pytest

from moeabench.mops import DPF2, DTLZ7
from moeabench.mops._dtlz7_pareto_regions import dtlz7_record_intervals
from scripts.audit.gt_disconnected_components_candidate_issue119 import (
    candidate, compare,
)


@pytest.mark.parametrize("cls,m,d,n",[
    (DTLZ7,3,None,32),(DTLZ7,5,None,128),
    (DTLZ7,10,None,128),(DTLZ7,10,None,512),
    (DPF2,5,3,64),(DPF2,10,9,128),(DPF2,10,9,512),
])
def test_disconnected_candidate_guarantees_optimal_component_occupancy_at_budget(cls,m,d,n):
    mop=cls(M=m) if cls is DTLZ7 else cls(M=m,D=d)
    decision,labels=candidate(mop,n)
    replay,relabels=candidate(mop,n)
    assert np.array_equal(decision,replay)
    assert np.array_equal(labels,relabels)
    assert decision.shape==(n,mop.N)
    assert np.all((decision>=0)&(decision<=1))
    q=m-1 if cls is DTLZ7 else d-1
    expected=2**q
    assert len(np.unique(labels))==min(n,expected)
    assert len(np.unique(mop.evaluation(decision)["F"],axis=0))==n
    assert np.all(decision[:,q:]==0)
    (lo0,hi0),(lo1,hi1)=dtlz7_record_intervals()
    position=decision[:,:q]
    assert np.all(((position>=lo0)&(position<=hi0))|
                  ((position>lo1)&(position<=hi1)))


@pytest.mark.parametrize("name,m,d",[
    ("DTLZ7",10,None),("DPF2",10,9),
])
def test_stratification_study_reports_coverage_without_certification(name,m,d):
    r=compare(name,m,d,128,probes=256)
    assert r["candidate_component_count"]==128
    assert r["candidate_max_possible_components"]==128
    assert r["incumbent_component_count"]<=128
    assert not r["budget_can_touch_all_components"]
    assert r["candidate"]["max_observed"]>=0
    assert r["incumbent"]["max_observed"]>=0
    assert r["finite_probe_not_continuous_certificate"]
    assert not r["public_MOP_mutated"]


def test_candidate_rejects_invalid_sample_budget():
    with pytest.raises(ValueError):
        candidate(DTLZ7(M=3),0)
    with pytest.raises(ValueError):
        candidate(DPF2(M=5,D=3),2.5)
