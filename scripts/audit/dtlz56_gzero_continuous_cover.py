# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact-rational component-only continuous covering certificate for DTLZ5/6.

Original M=4, K=1 g=0 Pareto curve. Every g=0 objective is a unit
vector F(theta)=cos(theta)*v + sin(theta)*e4 with ||v||=1, v.e4=0,
theta=pi*x1/2; all other angular positions collapse to pi/4.

For sorted EXACT dyadic first-position anchor coordinates 0=a0<...<ar=1,
each theta is within (pi/4)*max_i(a[i+1]-a[i]) of some anchor theta.
Unit-circle chord length <= angular separation. Machin upper rational pi
bound yields a rigorous *raw Euclidean* one-sided continuous curve-to-
reference distance upper bound. This does NOT certify g>0, arbitrary
whole-PF coverage, finite stored objective bits, normalized distances,
indicator suitability, or an ADR0059 release certificate.
"""
from __future__ import annotations

from fractions import Fraction
from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz56_exact_same_ray import _encode

PROOF_ID = "DTLZ56-M4K1-EXACT-GZERO-CURVE-COVER-V1"


def certify_gzero_curve_cover(decisions, *, problem, pi_terms=16):
    """Return conservative exact rational bound if 0/1 g=0 anchors verified."""
    if problem not in ("DTLZ5","DTLZ6"):
        raise ValueError("Only original DTLZ5/6 are supported")
    if type(pi_terms) is not int or not 4 <= pi_terms <= 32:
        raise ValueError("pi_terms must be an integer in [4,32]")
    try:
        rows = tuple(tuple(Fraction.from_float(float(v)) for v in row)
                     for row in decisions)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected finite four-dimensional decision rows") from exc
    if not rows or any(len(row)!=4 or any(not 0<=v<=1 for v in row)
                       for row in rows):
        raise ValueError("Expected feasible original M4 K1 decision matrix")

    target_tail = Fraction(1,2) if problem=="DTLZ5" else Fraction(0)
    coordinates=sorted(set(row[0] for row in rows if row[3]==target_tail))
    # At g=0, inner angular coordinates x2, x3 have no effect.
    valid=(len(coordinates)>=2 and coordinates[0]==0
           and coordinates[-1]==1)
    if not valid:
        return {
            "proof_id":PROOF_ID, "problem":problem,
            "conclusion":"unverified_missing_curve_endpoint",
            "anchor_x1": [_encode(v) for v in coordinates],
            "max_dyadic_parameter_gap":None,
            "raw_euclidean_cover_upper":None,
            "uniform_x1_mean_raw_distance_upper":None,
            "pi_terms":pi_terms,
            "continuous_gzero_component_only":True,
            "whole_gt_coverage":"unverified",
            "scientific_certification":"unverified",
        }

    max_gap=max(b-a for a,b in zip(coordinates,coordinates[1:]))
    pi_lo,pi_hi=_pi_rational_bounds(pi_terms)
    if not (3<pi_lo<pi_hi<4):
        raise ArithmeticError("Invalid rigorous Machin pi interval")
    upper=pi_hi*max_gap/4
    # Uniform measure in x1 on [0,1] is NOT an assumed measure over
    # the full original Pareto front. Integrate nearest-anchor angular
    # distance over each exact dyadic parameter gap: integral = gap²/4.
    uniform_mean_upper=pi_hi*sum(
        ((b-a)**2 for a,b in zip(coordinates,coordinates[1:])),
        Fraction(0),
    )/8
    return {
        "proof_id":PROOF_ID, "problem":problem,
        "conclusion":"rigorously_bounded_continuous_gzero_cover",
        "anchor_x1":[_encode(v) for v in coordinates],
        "max_dyadic_parameter_gap":_encode(max_gap),
        "raw_euclidean_cover_upper":_encode(upper),
        "uniform_x1_mean_raw_distance_upper":_encode(uniform_mean_upper),
        "pi_terms":pi_terms,
        "continuous_gzero_component_only":True,
        "whole_gt_coverage":"unverified",
        "scientific_certification":"unverified",
    }


def replay_gzero_curve_cover(decisions, record, *, problem, pi_terms=16):
    """Recompute all rational fields; never accept edited/unproved bounds."""
    try:
        expected=certify_gzero_curve_cover(
            decisions,problem=problem,pi_terms=pi_terms)
        return (
            record==expected and
            expected["conclusion"]==
            "rigorously_bounded_continuous_gzero_cover"
        )
    except (TypeError,ValueError,OverflowError,ArithmeticError):
        return False
