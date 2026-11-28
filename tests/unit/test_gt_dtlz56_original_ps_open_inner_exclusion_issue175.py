# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive source-decision exact angular-interior exclusion tests."""
from fractions import Fraction
import pytest

from scripts.audit.dtlz56_original_ps_open_inner_exclusion import (
    certify_original_ps_open_inner_exclusion,
    replay_original_ps_open_inner_exclusion,
)


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_open_inner_selected_rows_are_strictly_dominated(problem):
    tail_positive=.75 if problem=="DTLZ5" else .25
    tail_zero=.5 if problem=="DTLZ5" else 0.
    ps=[
        (.25,.25,.75,tail_positive), # provably dominated
        (.25,1.,.75,tail_positive),  # boundary: abstain
        (.25,.25,.75,tail_zero),     # g=0: abstain
        (.25,.25,0.,tail_positive),  # boundary: abstain
    ]
    proof=certify_original_ps_open_inner_exclusion(ps,problem=problem)
    assert replay_original_ps_open_inner_exclusion(
        ps,proof,problem=problem)
    assert proof["rigorously_dominated_open_inner_indices"]==[0]
    assert proof["rigorously_dominated_open_inner_count"]==1
    assert proof["whole_reference_membership"]=="rejected_has_dominated_member"
    assert proof["rows"][1]["conclusion"]=="unverified"
    assert proof["rows"][2]["conclusion"]=="unverified"
    assert proof["rows"][3]["conclusion"]=="unverified"
    assert proof["remaining_boundary_rows"]=="unverified"
    if problem=="DTLZ6":
        assert proof["rows"][0]["symbolic_irrational_g_witness"]
        assert proof["rows"][0]["exact_rational_witness"] is None
    else:
        assert proof["rows"][0]["exact_rational_witness"] is not None


def test_irrational_root_is_not_approximated_as_rational_g():
    ps=[(.25,.25,.75,1/3)]
    p=certify_original_ps_open_inner_exclusion(ps,problem="DTLZ6")
    assert p["rigorously_dominated_open_inner_indices"]==[0]
    assert replay_original_ps_open_inner_exclusion(
        ps,p,problem="DTLZ6")


def test_forged_and_altered_source_fail_replay():
    ps=[(.25,.25,.75,.75)]
    report=certify_original_ps_open_inner_exclusion(
        ps,problem="DTLZ5")
    forged=dict(report)
    forged["rigorously_dominated_open_inner_indices"]=[]
    assert not replay_original_ps_open_inner_exclusion(
        ps,forged,problem="DTLZ5")
    assert not replay_original_ps_open_inner_exclusion(
        [(.25,.25,.74,.75)],report,problem="DTLZ5")


@pytest.mark.parametrize("bad",[
    [],[(0.,0.,0.)],[(0.,0.,0.,float("nan"))],
    [(0.,0.,0.,1.1)],
])
def test_invalid_source_fails_closed(bad):
    with pytest.raises(ValueError):
        certify_original_ps_open_inner_exclusion(
            bad,problem="DTLZ6")


def test_boundary_only_source_does_not_get_false_approval():
    ps=[(0.,1.,0.,.5),(0.,1.,1.,.75)]
    result=certify_original_ps_open_inner_exclusion(
        ps,problem="DTLZ5")
    assert result["rigorously_dominated_open_inner_indices"]==[]
    assert result["whole_reference_membership"]=="unverified"
