# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact original M=4 K=1 DTLZ5/6 domination by the continuous g=0 PF curve.

Original g=0 objective image is (cos(a)/2,cos(a)/2,
cos(a)/sqrt(2),sin(a)), 0<=a<=pi/2. For an original
candidate y=(y1,y2,y3,c), 0<=c<=1, the exact g=0 PF
image with final objective c is
 (sqrt(1-c*c)/2,sqrt(1-c*c)/2,sqrt((1-c*c)/2),c).
It weakly dominates y precisely when
  4*y1*y1>=1-c*c; 4*y2*y2>=1-c*c;
  2*y3*y3>=1-c*c.
If g>0 then ||y||=1+g>1 so at least one inequality
must be strict; hence Pareto domination is strict.
This is a genuine OFF-RAY exact-real global dominance
condition and applies irrespective of the angular boundary pattern.

To rigorously certify from IEEE-754 original decisions, use
outward EXACT-RATIONAL interval enclosures of each of the four
original trigonometric objective images, including integer tenth-root
enclosures for irrational DTLZ6 g. The sufficient interval test uses
the LOWER rational bounds on y1,y2,y3 and c, and an UPPER rational
bound on c to prove 0<=c<=1. It deliberately abstains on all
inconclusive/near-boundary cases and NEVER certifies PF membership.
"""
from __future__ import annotations

from fractions import Fraction

from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz5_patch_actual_source_cover import (
    _exact_decisions,_endpoint_safe_objective_intervals,
)
from .dtlz6_patch_actual_source_cover import _outward_original_objectives
from .dtlz56_exact_same_ray import _encode

SCHEMA="DTLZ56-M4K1-GZERO-OFF-RAY-CURVE-DOMINANCE-V1"


def certify_gzero_curve_offray_dominance(decision,*,problem,
                                         root_bits=56,pi_terms=16):
    if problem not in ("DTLZ5","DTLZ6"):
        raise ValueError("Expected original M4 K1 DTLZ5/6")
    if type(root_bits) is not int or not 8<=root_bits<=80:
        raise ValueError("Expected integer tenth-root precision")
    if type(pi_terms) is not int or not 4<=pi_terms<=32:
        raise ValueError("Expected supported Machin precision")
    (row,)=_exact_decisions([decision])
    pi=_pi_rational_bounds(pi_terms)
    if problem=="DTLZ5":
        intervals=_endpoint_safe_objective_intervals(row,pi)
        strictly_positive_g=row[-1]!=Fraction(1,2)
    else:
        intervals=_outward_original_objectives(row,pi,root_bits=root_bits)
        strictly_positive_g=row[-1]>0
    if len(intervals)!=4 or any(
        not 0<=lo<=hi for lo,hi in intervals
    ):
        raise ArithmeticError("Invalid outward original objective intervals")
    c_lo,c_hi=intervals[3]
    cap=c_hi<=1
    right=1-c_lo*c_lo
    factors=(4,4,2)
    margins=tuple(Fraction(k)*intervals[i][0]**2-right
                  for i,k in enumerate(factors))
    strict=bool(strictly_positive_g and cap and all(m>=0 for m in margins))
    return {
        "schema":SCHEMA,"problem":problem,"M":4,"K":1,
        "original_decision":[_encode(x) for x in row],
        "original_objective_intervals":[
            [_encode(lo),_encode(hi)] for lo,hi in intervals
        ],
        "candidate_g_strictly_positive":strictly_positive_g,
        "rigorous_last_objective_le_one":cap,
        "squared_coordinate_margins_lower":[_encode(m) for m in margins],
        "gzero_exact_mathematical_dominator":
            "(sqrt(1-c^2)/2,sqrt(1-c^2)/2,sqrt((1-c^2)/2),c)",
        "conclusion":(
            "rigorously_strictly_dominated_by_gzero_curve"
            if strict else "unverified"
        ),
        "offray_global_pf_membership":"unverified",
        "whole_gt_coverage":"unverified",
        "scientific_certification":"unverified",
    }


def replay_gzero_curve_offray_dominance(decision,record,*,problem,
                                        root_bits=56,pi_terms=16):
    try:
        expected=certify_gzero_curve_offray_dominance(
            decision,problem=problem,root_bits=root_bits,
            pi_terms=pi_terms)
        return record==expected and expected["conclusion"]==(
            "rigorously_strictly_dominated_by_gzero_curve")
    except (TypeError,ValueError,KeyError,ArithmeticError,
            OverflowError,ZeroDivisionError):
        return False
