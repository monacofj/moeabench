# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rigorous last-objective saturation exclusion in original DTLZ5/6.

For M>=2 original DTLZ5/6, F_M=(1+g)sin(pi*x1/2).
The ORIGINAL feasible global reference decision (x1=1,g=0,
all inner positions arbitrary, with feasible g=0 tail) has exact
objective image (0,...,0,1). Hence if F_M>1, that point
STRICTLY dominates any given original feasible decision, with
no conditions on its inner angles. Independently, x1=1,g>0
is strictly dominated even when the Taylor sine lower interval
cannot prove it, since exact sin(pi/2)=1.

Rigorous sine LOWER interval uses the independent rational
Machin/Taylor enclosure, never production floating trig.
A supplied rational g is an exact declaration, or a separate
rigorous rational LOWER bound for possibly irrational g
(e.g. original DTLZ6 tail tenth root); using g_lower <= real g
preserves all strict domination conclusions. Unresolved
cases ABSTAIN, never establish front membership.
"""
from __future__ import annotations
from fractions import Fraction

from .dtlz56_rational_interval import (
    _as_exact_fraction,_pi_rational_bounds,
)
from .dtlz5_row10_exact_dominance import _trig_bounds
from .dtlz56_exact_same_ray import _encode

SCHEMA="DTLZ56-LAST-OBJECTIVE-UNIT-CAP-EXCLUSION-V1"


def certify_last_objective_cap_exclusion(
    x1, *,g_lower,problem,k,pi_terms=16,
):
    if problem not in ("DTLZ5","DTLZ6") or type(k) is not int or k<1:
        raise ValueError("Expected original DTLZ5/6 with K>=1")
    if type(pi_terms) is not int or not 4<=pi_terms<=32:
        raise ValueError("Expected rigorously supported pi_terms")
    x=_as_exact_fraction(x1,name="x1")
    lo=_as_exact_fraction(g_lower,name="g_lower")
    cap=Fraction(k,4) if problem=="DTLZ5" else Fraction(k)
    if not 0<=x<=1 or not 0<=lo<=cap:
        raise ValueError("Expected feasible x1 and lower radial bound")
    pi=_pi_rational_bounds(pi_terms)
    sin_lo=_trig_bounds(x/2,pi,sine=True)[0]
    if not 0<=sin_lo<=1:
        raise ArithmeticError("Expected valid nonnegative outward sine lower")
    margin=(1+lo)*sin_lo-1
    # Symbolic endpoint exclusion even for arbitrarily small g>0,
    # beyond any finite Taylor enclosure ability to separate from 1.
    endpoint=(x==1 and lo>0)
    proved=endpoint or margin>0
    return {
        "proof_id":SCHEMA,"problem":problem,"K":k,
        "x1":_encode(x),"radial_g_rational_lower":_encode(lo),
        "pi_terms":pi_terms,
        "sin_original_last_angle_lower":_encode(sin_lo),
        "strict_last_objective_separation_lower":_encode(margin),
        "exact_x1_one_endpoint":endpoint,
        "conclusion":(
            "rigorously_strictly_dominated_by_gzero_last_axis"
            if proved else "unverified"
        ),
        "exact_dominator_objective":
            "(0,...,0,1); original feasible x1=1,g=0",
        "candidate_g_may_be_irrational":True,
        "boundary_or_remaining_pf_membership":"unverified",
        "scientific_certification":"unverified",
    }


def replay_last_objective_cap_exclusion(
    x1,record,*,g_lower,problem,k,pi_terms=16,
):
    try:
        expected=certify_last_objective_cap_exclusion(
            x1,g_lower=g_lower,problem=problem,k=k,
            pi_terms=pi_terms)
        return (record==expected and expected["conclusion"]==
                "rigorously_strictly_dominated_by_gzero_last_axis")
    except (TypeError,ValueError,ArithmeticError,AssertionError,
            KeyError,OverflowError,ZeroDivisionError):
        return False
