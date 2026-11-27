# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact necessary boundary condition for original DTLZ5/DTLZ6 positive-g PF.

For original M>=3, K>=1 and g>0, EVERY decision with ALL inner
positions x2,...,x_(M-1) STRICTLY inside (0,1) is dominated on its
own objective ray. Therefore a positive-g Pareto decision must have
at least one inner coordinate on {0,1}. In original M=4, this
reduces the still-open candidate domain to the four angular
boundary faces x2=0, x2=1, x3=0, or x3=1 (with intersections).
It does NOT prove that any given boundary point is Pareto optimal,
nor that the existing two sufficient supporting surfaces exhaust
the positive-g Pareto front.

Proof: original theta_i=pi/4*(1+g*(2xi-1)/(1+g)) for i>=2.
The angular inner deviations d_i=g*(2xi-1)/(1+g) satisfy
q=max|d_i| < g/(1+g) if every xi in (0,1). Thus
gmin=q/(1-q)<g; choose gp=(g+gmin)/2 and set
x_i'=1/2+d_i*(1+gp)/(2gp) in [0,1].
All original objective angles stay EXACTLY equal, while
F'=(1+gp)/(1+g)*F with strict scale <1. Since objectives
are nonnegative and their squared sum is (1+g)^2>0, this
proves strict Pareto dominance (all <= and some strictly <).
Feasible original radial tails for gp exist (DTLZ5 g in
[0,K/4] and DTLZ6 g in [0,K]). No production or float math.
"""
from __future__ import annotations
from fractions import Fraction

from .dtlz56_rational_interval import _as_exact_fraction
from .dtlz56_exact_same_ray import (
    _encode,construct_same_ray_witness,verify_same_ray_witness,
)

SCHEMA="DTLZ56-POSITIVE-G-OPEN-INNER-ANGLES-EXCLUSION-V1"


def certify_open_inner_angle_exclusion(positions,*,g,problem,k):
    """Exact conditional open-interior dominance witness; otherwise abstain."""
    if problem not in ("DTLZ5","DTLZ6") or type(k) is not int or k<1:
        raise ValueError("Expected original DTLZ5/6 integer K>=1")
    try:
        u=tuple(_as_exact_fraction(v,name="position") for v in positions)
        radial=_as_exact_fraction(g,name="g")
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("Expected finite exact-rational declarations") from exc
    if len(u)<2 or any(not 0<=v<=1 for v in u):
        raise ValueError("Expected M>=3 and feasible angular positions")
    gmax=Fraction(k,4) if problem=="DTLZ5" else Fraction(k)
    if not 0<=radial<=gmax:
        raise ValueError("Expected original feasible g range")
    inside=all(0<v<1 for v in u[1:])
    witnessed=None
    if radial>0 and inside:
        witness=construct_same_ray_witness(
            u,g=radial,problem=problem,k=k)
        if (witness.conclusion!="exact_same_ray_dominator"
            or not verify_same_ray_witness(
                u,witness.to_dict(),g=radial,problem=problem,k=k)):
            raise AssertionError("Open-inner-angle exclusion proof did not replay")
        witnessed=witness.to_dict()
    return {
        "proof_id":SCHEMA,
        "problem":problem,"M":len(u)+1,"K":k,
        "positions":[_encode(v) for v in u],
        "radial_g":_encode(radial),
        "all_inner_coordinates_open":inside,
        "conclusion":(
            "rigorously_dominated_open_inner_angular_region"
            if witnessed is not None else "unverified"
        ),
        "same_ray_exact_dominator":witnessed,
        "necessary_positive_g_pf_condition":
            "at least one inner x_i is exactly 0 or 1",
        "universal_boundary_exclusion_theorem":True,
        "sufficient_boundary_membership":"unverified",
        "whole_pf_completeness":"unverified",
        "scientific_certification":"unverified",
    }


def replay_open_inner_angle_exclusion(
    positions,record,*,g,problem,k,
):
    try:
        expected=certify_open_inner_angle_exclusion(
            positions,g=g,problem=problem,k=k)
        return (record==expected and expected["conclusion"]==
                "rigorously_dominated_open_inner_angular_region")
    except (TypeError,ValueError,AssertionError,ArithmeticError,
            OverflowError,ZeroDivisionError):
        return False
