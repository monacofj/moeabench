# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact global-PF theorem and cover for FULL RADIAL face x2=x3=1.

Original M=4 K=1 DTLZ5: x1 in [0,1/4], x2=x3=1, x4=t in [0,1].
Original M=4 K=1 DTLZ6: x1 in [0,3/16], x2=x3=1, x4=u**10,
                      u=g in [0,1].

These are *continuous* (x1, radial) subsets, including ALL original
allowed radial magnitudes and two tail branches of DTLZ5. The original
Taylor-sine p=2 global supporting theorem is uniform, because
 ((5/4)*pi_upper/8)^2 < 1-2/pi_lower   (DTLZ5)
 (2*pi_upper*(3/16)/2)^2 < 1-2/pi_lower (DTLZ6).
The exact uniform proof margin must be positive.

Lipschitz on either patch:
 ||dF/dx1|| <= rmax*pi/2,
 ||dF/dt||  <= 1+pi/2  (DTLZ5; |dg/dt|<=1),
 ||dF/du||  <= 1+pi/2  (DTLZ6; g=u).
Thus an exact tensor grid with maximum gaps dx,dr covers the entire
continuous 2D subset in raw Euclidean norm by
 rmax*pi_upper*dx/4+(1+pi_upper/2)*dr/2.

The grid consists of exact rational DECISION declarations; some tail
values are only representable as exact rationals (not IEEE-754).
No production sampler, whole PF, stored F, IGD/HV claim.
"""
from __future__ import annotations

from fractions import Fraction

from .dtlz56_exact_same_ray import _encode
from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz56_tight_sine_global_pf import (
    certify_tight_sine_global_pf,replay_tight_sine_global_pf,
)
from .dtlz56_dtlz6_radial_enclosure import (
    certify_dtlz6_radial_interval_pf,replay_dtlz6_radial_interval_pf,
)

SCHEMA="dtlz56_m4k1_full_radial_support_face_v1"


def _axis(axis, last):
    if not isinstance(axis,(tuple,list)) or len(axis)<2:
        raise ValueError("Require endpoints and at least two exact grid nodes")
    try:
        numbers=tuple(Fraction(x) if isinstance(x,Fraction)
                      else Fraction.from_float(float(x))
                      for x in axis)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError("Invalid exact rational grid axis") from exc
    if (numbers[0]!=0 or numbers[-1]!=last or
        any(not 0<=a<b<=last for a,b in zip(numbers,numbers[1:]))):
        raise ValueError("Grid must increase strictly between exact endpoints")
    return numbers


def certify_full_radial_support_face(
    *,problem,
    x1_grid=None,radial_grid=None,pi_terms=16,
):
    if problem not in ("DTLZ5","DTLZ6"):
        raise ValueError("Only original DTLZ5/6 M4 K1")
    if type(pi_terms) is not int or not 4<=pi_terms<=32:
        raise ValueError("Invalid rigorous pi terms")
    is_five=problem=="DTLZ5"
    xmax=Fraction(1,4) if is_five else Fraction(3,16)
    radialmax=Fraction(1)  # t in [0,1] or u in [0,1]
    rmax=Fraction(5,4) if is_five else Fraction(2)
    if x1_grid is None:
        x1_grid=tuple(xmax*i/4 for i in range(5))
    if radial_grid is None:
        radial_grid=tuple(Fraction(i,4) for i in range(5))
    xs=_axis(x1_grid,xmax)
    rs=_axis(radial_grid,radialmax)
    pi_lo,pi_hi=_pi_rational_bounds(pi_terms)
    if not 3<pi_lo<pi_hi<4:
        raise ArithmeticError("Rigorous pi enclosure required")
    sine_squared=(rmax*pi_hi*xmax/2)**2
    threshold=1-Fraction(2,1)/pi_lo
    strict_margin=threshold-sine_squared
    if strict_margin<=0:
        raise AssertionError("Full radial support-face PF proof failed")
    dx=max(b-a for a,b in zip(xs,xs[1:]))
    dr=max(b-a for a,b in zip(rs,rs[1:]))
    raw_cover=rmax*pi_hi*dx/4+(1+pi_hi/2)*dr/2
    nodes=[]
    for x in xs:
        for r in rs:
            tail=r if is_five else r**10
            g=(tail-Fraction(1,2))**2 if is_five else r
            decision=(x,Fraction(1),Fraction(1),tail)
            if g==0:
                kind="exact_unit_radius"
            elif is_five:
                proof=certify_tight_sine_global_pf(
                    decision[:3],g=g,problem=problem,k=1)
                if not replay_tight_sine_global_pf(
                    decision[:3],proof.to_dict(),g=g,problem=problem,k=1):
                    raise AssertionError("Positive DTLZ5 PF replay failed")
                kind="exact_sine_power_pf"
            else:
                proof=certify_dtlz6_radial_interval_pf(
                    decision[:3],decision[3:],
                    k=1,root_bits=56,pi_terms=16)
                if not replay_dtlz6_radial_interval_pf(
                    decision[:3],decision[3:],proof.to_dict(),
                    k=1,root_bits=56,pi_terms=16):
                    raise AssertionError("Positive DTLZ6 PF replay failed")
                kind="exact_tenth_root_sine_power_pf"
            nodes.append({
                "decision":[_encode(v) for v in decision],
                "radial_g":_encode(g),
                "positive_pf_proof":kind,
            })
    return {
        "schema":SCHEMA,
        "problem":problem,"M":4,"K":1,
        "conclusion":"rigorously_global_pf_continuous_full_radial_face",
        "x1_domain":[_encode(Fraction(0)),_encode(xmax)],
        "radial_domain":[_encode(Fraction(0)),_encode(radialmax)],
        "radial_parameter":("x4=t" if is_five else "g=u,x4=u^10"),
        "support_x2_x3":[_encode(Fraction(1)),_encode(Fraction(1))],
        "x1_grid":[_encode(v) for v in xs],
        "radial_grid":[_encode(v) for v in rs],
        "global_pf_strict_uniform_margin":_encode(strict_margin),
        "continuous_to_exact_grid_raw_l2_upper":_encode(raw_cover),
        "grid_decisions":nodes,
        "certified_full_original_radial_range":True,
        "restricted_angular_support_face_only":True,
        "actual_selected_source_alignment":"unverified",
        "whole_positive_g_pf":"unverified",
        "indicator_fitness":"unverified",
        "scientific_certification":"unverified",
    }


def replay_full_radial_support_face(
    record,*,problem,x1_grid=None,radial_grid=None,pi_terms=16,
):
    try:
        expected=certify_full_radial_support_face(
            problem=problem,x1_grid=x1_grid,radial_grid=radial_grid,
            pi_terms=pi_terms)
        return (record==expected and
                expected["conclusion"]==
                "rigorously_global_pf_continuous_full_radial_face")
    except (ValueError,TypeError,ArithmeticError,AssertionError,
            ZeroDivisionError,OverflowError):
        return False
