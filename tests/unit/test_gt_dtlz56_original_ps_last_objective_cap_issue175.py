# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive last-objective cap original PS proof replay and abstention."""
import pytest
from scripts.audit.dtlz56_original_ps_last_objective_cap import (
    certify_original_ps_last_objective_cap,
    replay_original_ps_last_objective_cap,
)


@pytest.mark.parametrize("problem,tail",[
    ("DTLZ5",.875),("DTLZ6",.25),
])
def test_source_last_cap_strictly_dominated_with_boundary_angles(problem,tail):
    ps=[
        (.75,1.,0.,tail),
        (.25,1.,0.,tail),
        (1.,1.,1.,tail),
    ]
    proof=certify_original_ps_last_objective_cap(ps,problem=problem)
    assert replay_original_ps_last_objective_cap(ps,proof,problem=problem)
    assert proof["provably_dominated_last_objective_rows"]==[0,2]
    assert proof["whole_gt_membership"]=="rejected_has_dominated_member"
    assert proof["rows"][1]["last_objective_proof"]["conclusion"]=="unverified"
    assert proof["scientific_certification"]=="unverified"


def test_exact_gzero_source_endpoint_abstains():
    ps=[(1.,1.,1.,.5),(1.,1.,1.,0.)]
    r=certify_original_ps_last_objective_cap(ps,problem="DTLZ5")
    assert r["provably_dominated_last_objective_rows"]==[1]
    assert r["rows"][0]["last_objective_proof"]["conclusion"]=="unverified"


def test_small_irrational_radial_lower_supported():
    ps=[(1.,1.,1.,.3333333333333333)]
    r=certify_original_ps_last_objective_cap(ps,problem="DTLZ6")
    assert r["provably_dominated_last_objective_rows"]==[0]


def test_forged_proof_and_different_ps_rejected():
    ps=[(1.,1.,1.,.25)]
    proof=certify_original_ps_last_objective_cap(ps,problem="DTLZ6")
    forged=dict(proof)
    forged["provably_dominated_last_objective_rows"]=[]
    assert not replay_original_ps_last_objective_cap(
        ps,forged,problem="DTLZ6")
    assert not replay_original_ps_last_objective_cap(
        [(1.,1.,1.,.5)],proof,problem="DTLZ6")


@pytest.mark.parametrize("bad",[
    [],[(0.,0.,0.)],[(0.,0.,0.,float("nan"))],
    [(0.,0.,0.,1.01)],
])
def test_invalid_source_fails_closed(bad):
    with pytest.raises(ValueError):
        certify_original_ps_last_objective_cap(bad,problem="DTLZ5")
