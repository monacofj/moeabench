# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact full-radial second support stratum for original DTLZ5/6 M4 K1.

Fix (x1,x2,x3)=(0,1,0), and vary the COMPLETE radial domain:
DTLZ5 x4=t in [0,1]; DTLZ6 x4=u**10, u=g in [0,1].
The independent generalized sine-power theorem proves each entire
continuous decision-image curve globally Pareto-optimal, because its
j=2 supporting component is h_2(g) decreasing in g.

Original spherical unit-vector derivative satisfies ||dU/dg||<=pi/(2r);
||dF/dg||<=1+pi/2. For DTLZ5 |dg/dt|=2|t-.5|<=1;
for DTLZ6 g=u. An exact grid with radial max gap delta covers the
continuous curve within (1+pi_upper/2)*delta/2 in raw L2.

A separate exact outward objective interval step bounds each node's
distance to the ACTUAL original selected PS. Neither fact covers all PF.
"""
from __future__ import annotations

from fractions import Fraction

from .dtlz56_generalized_positive_g_pf import (
    certify_generalized_positive_g_pf,
    replay_generalized_positive_g_pf,
)
from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz56_exact_same_ray import _encode

SCHEMA="dtlz56_m4k1_second_support_full_radial_curve_v1"
POSITIONS=(Fraction(0),Fraction(1),Fraction(0))


def _axis(values):
    if not isinstance(values,(tuple,list)) or len(values)<2:
        raise ValueError("At least two radial endpoints required")
    try:
        xs=tuple(Fraction(v) if isinstance(v,Fraction)
                 else Fraction.from_float(float(v)) for v in values)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError("Invalid radial coordinates") from exc
    if xs[0]!=0 or xs[-1]!=1 or any(
        not 0<=a<b<=1 for a,b in zip(xs,xs[1:])
    ):
        raise ValueError("Must strictly cover radial parameter [0,1]")
    return xs


def certify_second_support_full_radial(*,problem,radial_grid=None,pi_terms=16):
    if problem not in ("DTLZ5","DTLZ6"):
        raise ValueError("Only original DTLZ5/6")
    if type(pi_terms) is not int or not 4<=pi_terms<=32:
        raise ValueError("Invalid Machin pi precision")
    axis=_axis(radial_grid if radial_grid is not None
               else tuple(Fraction(i,8) for i in range(9)))
    pi_lo,pi_hi=_pi_rational_bounds(pi_terms)
    if not 3<pi_lo<pi_hi<4:
        raise ArithmeticError("No rigorous pi bounds")
    delta=max(b-a for a,b in zip(axis,axis[1:]))
    covering=(1+pi_hi/2)*delta/2
    nodes=[]
    for radial in axis:
        tail=radial if problem=="DTLZ5" else radial**10
        g=(radial-Fraction(1,2))**2 if problem=="DTLZ5" else radial
        proof=certify_generalized_positive_g_pf(
            POSITIONS,g=g,problem=problem,k=1)
        if proof.conclusion!="rigorously_global_pf_member" or (
            not replay_generalized_positive_g_pf(
                POSITIONS,proof.to_dict(),g=g,problem=problem,k=1
            )
        ):
            raise AssertionError("Exact generalized support proof failed")
        if proof.supporting_objective!=2 or proof.sine_power!=2:
            raise AssertionError("Wrong supporting objective or exponent")
        nodes.append({
            "decision":[_encode(v) for v in (*POSITIONS,tail)],
            "radial_g":_encode(g),
            "certificate":proof.to_dict(),
        })
    return {
        "schema":SCHEMA,"problem":problem,"M":4,"K":1,
        "conclusion":"rigorously_global_pf_complete_second_radial_curve",
        "positions":[_encode(v) for v in POSITIONS],
        "radial_grid":[_encode(v) for v in axis],
        "radial_parameter":"x4=t" if problem=="DTLZ5" else "g=u,x4=u^10",
        "complete_admissible_g_range":True,
        "radial_max_gap":_encode(delta),
        "raw_l2_continuous_to_grid_upper":_encode(covering),
        "grid_decisions":nodes,
        "second_support_stratum_only":True,
        "other_pf_strata":"unverified",
        "whole_gt_coverage":"unverified",
        "scientific_certification":"unverified",
    }


def replay_second_support_full_radial(record,*,problem,radial_grid=None,pi_terms=16):
    try:
        correct=certify_second_support_full_radial(
            problem=problem,radial_grid=radial_grid,pi_terms=pi_terms)
        return record==correct and (
            correct["conclusion"]=="rigorously_global_pf_complete_second_radial_curve"
        )
    except (ValueError,TypeError,ArithmeticError,AssertionError,OverflowError):
        return False
