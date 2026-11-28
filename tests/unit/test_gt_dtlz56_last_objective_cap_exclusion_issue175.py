# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive exact last-objective cap dominance theorem checks."""
from fractions import Fraction
import pytest

from scripts.audit.dtlz56_last_objective_cap_exclusion import (
    certify_last_objective_cap_exclusion,
    replay_last_objective_cap_exclusion,
)
from scripts.audit.dtlz56_exact_same_ray import _decode


@pytest.mark.parametrize("problem,k,g",[
    ("DTLZ5",1,Fraction(1,8)),
    ("DTLZ6",1,Fraction(1,8)),
    ("DTLZ5",2,Fraction(1,4)),
    ("DTLZ6",2,Fraction(1,4)),
])
def test_large_first_angle_and_positive_g_excluded(problem,k,g):
    x=Fraction(3,4)
    p=certify_last_objective_cap_exclusion(
        x,g_lower=g,problem=problem,k=k)
    assert p["conclusion"]=="rigorously_strictly_dominated_by_gzero_last_axis"
    assert replay_last_objective_cap_exclusion(
        x,p,g_lower=g,problem=problem,k=k)
    assert _decode(p["strict_last_objective_separation_lower"])>0


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_exact_endpoint_excludes_arbitrarily_small_positive_g(problem):
    g=Fraction(1,10**50)
    p=certify_last_objective_cap_exclusion(
        1,g_lower=g,problem=problem,k=1)
    assert p["exact_x1_one_endpoint"] is True
    assert replay_last_objective_cap_exclusion(
        1,p,g_lower=g,problem=problem,k=1)


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_gzero_and_low_last_objective_abstain(problem):
    for x,g in ((Fraction(1,2),0),(0,Fraction(1,10)),
                (Fraction(1,2),Fraction(1,10))):
        p=certify_last_objective_cap_exclusion(
            x,g_lower=g,problem=problem,k=1)
        assert p["conclusion"]=="unverified"
        assert not replay_last_objective_cap_exclusion(
            x,p,g_lower=g,problem=problem,k=1)


def test_irrational_radial_g_can_use_rigorous_rational_lower():
    # g=tail**(1/10) may be irrational; a certified g_lower is sufficient.
    p=certify_last_objective_cap_exclusion(
        Fraction(3,4),g_lower=Fraction(1,8),problem="DTLZ6",k=1)
    assert replay_last_objective_cap_exclusion(
        Fraction(3,4),p,g_lower=Fraction(1,8),problem="DTLZ6",k=1)


def test_forged_or_different_witness_rejected():
    g=Fraction(1,8)
    p=certify_last_objective_cap_exclusion(
        Fraction(3,4),g_lower=g,problem="DTLZ5",k=1)
    bad=dict(p)
    bad["strict_last_objective_separation_lower"]={
        "numerator_hex":"0x1","denominator_hex":"0x1"}
    assert not replay_last_objective_cap_exclusion(
        Fraction(3,4),bad,g_lower=g,problem="DTLZ5",k=1)
    assert not replay_last_objective_cap_exclusion(
        Fraction(7,8),p,g_lower=g,problem="DTLZ5",k=1)


@pytest.mark.parametrize("x,g,problem,k,terms",[
    (-.1,.1,"DTLZ5",1,16),
    (1.1,.1,"DTLZ6",1,16),
    (.5,-.1,"DTLZ5",1,16),
    (.5,.26,"DTLZ5",1,16),
    (.5,float("nan"),"DTLZ6",1,16),
    (.5,.1,"DTLZ7",1,16),
    (.5,.1,"DTLZ5",0,16),
    (.5,.1,"DTLZ5",1,1),
])
def test_invalid_declaration_fails_closed(x,g,problem,k,terms):
    with pytest.raises(ValueError):
        certify_last_objective_cap_exclusion(
            x,g_lower=g,problem=problem,k=k,pi_terms=terms)
