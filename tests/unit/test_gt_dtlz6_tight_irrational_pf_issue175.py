# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New additive exact DTLZ6 Taylor/root PF membership tests."""
from fractions import Fraction
import pytest
from scripts.audit.dtlz6_tight_irrational_pf import (
    certify_dtlz6_tight_radial_pf,replay_dtlz6_tight_radial_pf,
)
from scripts.audit.dtlz6_tight_selected_source import (
    audit_dtlz6_tight_selected,replay_dtlz6_tight_selected,
)
from scripts.audit.dtlz56_exact_same_ray import _decode


@pytest.mark.parametrize("tail",[
    Fraction(1,16),Fraction(1,4),Fraction(1,3),Fraction(3,4),
])
def test_proved_boundary_rational_or_irrational_g(tail):
    u=(Fraction(0),Fraction(1),Fraction(0))
    r=certify_dtlz6_tight_radial_pf(u,(tail,),k=1)
    assert r["conclusion"]=="rigorously_global_pf_member"
    assert replay_dtlz6_tight_radial_pf(u,(tail,),r,k=1)
    assert r["supporting_objective"]==2
    assert _decode(r["strict_rational_margin"])>0
    assert r["scientific_certification"]=="unverified"


def test_new_taylor_upper_never_worse_than_coarse_pi_angle():
    u=(Fraction(1,8),Fraction(1),Fraction(0))
    record=certify_dtlz6_tight_radial_pf(
        u,(Fraction(1,3),),k=1)
    assert record["conclusion"]=="rigorously_global_pf_member"
    assert replay_dtlz6_tight_radial_pf(
        u,(Fraction(1,3),),record,k=1)


def test_gzero_and_off_pattern_abstain():
    for u,tail in (
        ((Fraction(0),Fraction(1),Fraction(0)),Fraction(0)),
        ((Fraction(1,8),Fraction(1,2),Fraction(1,2)),Fraction(1,3)),
    ):
        r=certify_dtlz6_tight_radial_pf(u,(tail,),k=1)
        assert r["conclusion"]=="unverified"
        assert not replay_dtlz6_tight_radial_pf(u,(tail,),r,k=1)


def test_selected_ledger_and_tamper_rejection():
    ps=[
        (0.,1.,0.,1/3),
        (.125,1.,0.,.25),
        (.5,.5,.5,0.),
        (.5,.5,.5,.25),
    ]
    r=audit_dtlz6_tight_selected(ps)
    assert replay_dtlz6_tight_selected(ps,r)
    assert r["source_selected_n"]==4
    assert set(r["resolved_historical_unverified_indices"]).issubset(
        set(r["historical_unverified_indices"]))
    assert set(r["still_unverified_historical_indices"])==(
        set(r["historical_unverified_indices"])-
        set(r["resolved_historical_unverified_indices"]))
    assert r["proved_selected_count"]>=3
    assert 3 in r["still_unverified_indices"]
    assert r["scientific_certification"]=="unverified"
    forged=dict(r)
    forged["still_unverified_indices"]=[]
    assert not replay_dtlz6_tight_selected(ps,forged)


@pytest.mark.parametrize("positions,tail,k",[
    ((0.,1.,0.),(float("nan"),),1),
    ((0.,1.,0.),(.25,.25),1),
    ((0.,1.1,0.),(.25,),1),
    ((0.,1.,0.),(.25,),0),
])
def test_invalid_input_fails_closed(positions,tail,k):
    with pytest.raises(ValueError):
        certify_dtlz6_tight_radial_pf(positions,tail,k=k)
