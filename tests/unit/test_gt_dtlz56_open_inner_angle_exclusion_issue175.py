# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New additive exact exclusion theorem tests, no historic regressions edited."""
from fractions import Fraction
import pytest

from scripts.audit.dtlz56_open_inner_angle_exclusion import (
    certify_open_inner_angle_exclusion,replay_open_inner_angle_exclusion,
)
from scripts.audit.dtlz56_exact_same_ray import _decode


@pytest.mark.parametrize("problem,g",[
    ("DTLZ5",Fraction(1,8)),("DTLZ6",Fraction(3,5)),
])
@pytest.mark.parametrize("x2,x3",[
    (Fraction(1,2),Fraction(1,2)),
    (Fraction(1,4),Fraction(3,4)),
    (Fraction(1,2**20),Fraction(2**20-1,2**20)),
])
def test_every_open_inner_face_has_exact_global_dominator(problem,g,x2,x3):
    u=(Fraction(1,3),x2,x3)
    proof=certify_open_inner_angle_exclusion(
        u,g=g,problem=problem,k=1)
    assert proof["conclusion"]=="rigorously_dominated_open_inner_angular_region"
    assert replay_open_inner_angle_exclusion(
        u,proof,g=g,problem=problem,k=1)
    witness=proof["same_ray_exact_dominator"]
    assert _decode(witness["witness_g"])<g
    assert 0<_decode(witness["objective_scale"])<1
    assert proof["sufficient_boundary_membership"]=="unverified"
    assert proof["whole_pf_completeness"]=="unverified"


@pytest.mark.parametrize("inner",[
    (Fraction(0),Fraction(1,2)),
    (Fraction(1),Fraction(1,2)),
    (Fraction(1,2),Fraction(0)),
    (Fraction(1,2),Fraction(1)),
])
def test_boundary_is_abstention_not_false_pareto_claim(inner):
    u=(Fraction(1,4),*inner)
    p=certify_open_inner_angle_exclusion(
        u,g=Fraction(1,8),problem="DTLZ5",k=1)
    assert p["conclusion"]=="unverified"
    assert not replay_open_inner_angle_exclusion(
        u,p,g=Fraction(1,8),problem="DTLZ5",k=1)


def test_g_zero_is_not_excluded():
    u=(Fraction(1,5),Fraction(1,2),Fraction(1,2))
    p=certify_open_inner_angle_exclusion(u,g=0,problem="DTLZ5",k=1)
    assert p["conclusion"]=="unverified"
    assert not replay_open_inner_angle_exclusion(
        u,p,g=0,problem="DTLZ5",k=1)


def test_forged_witness_and_wrong_g_fail_closed():
    u=(Fraction(1,3),Fraction(1,2),Fraction(1,2))
    p=certify_open_inner_angle_exclusion(
        u,g=Fraction(1,4),problem="DTLZ5",k=1)
    forged=dict(p)
    forged["same_ray_exact_dominator"]=None
    assert not replay_open_inner_angle_exclusion(
        u,forged,g=Fraction(1,4),problem="DTLZ5",k=1)
    assert not replay_open_inner_angle_exclusion(
        u,p,g=Fraction(1,8),problem="DTLZ5",k=1)


def test_generic_m5_and_k2():
    u=(Fraction(2,5),Fraction(1,2),Fraction(1,3),Fraction(2,3))
    p=certify_open_inner_angle_exclusion(
        u,g=Fraction(2,5),problem="DTLZ5",k=2)
    assert p["M"]==5
    assert replay_open_inner_angle_exclusion(
        u,p,g=Fraction(2,5),problem="DTLZ5",k=2)


@pytest.mark.parametrize("positions,g,problem,k",[
    ((.5,.5,.5),-.1,"DTLZ5",1),
    ((.5,.5,.5),.3,"DTLZ5",1),
    ((.5,.5,.5),float("nan"),"DTLZ6",1),
    ((.5,1.2,.5),.3,"DTLZ6",1),
    ((.5,.5,.5),.3,"DTLZ7",1),
    ((.5,.5,.5),.3,"DTLZ5",0),
])
def test_invalid_declarations_rejected(positions,g,problem,k):
    with pytest.raises(ValueError):
        certify_open_inner_angle_exclusion(
            positions,g=g,problem=problem,k=k)
