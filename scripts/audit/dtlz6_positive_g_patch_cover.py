# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Certified positive-g continuous PF patch for original DTLZ6 M4/K1.

Use u=g as the radial parameter and x4=u**10, not x4 directly:
original DTLZ6 has g=(x4)**(1/10). On the continuous patch
  x1 in [0,1/8], x2=x3=1, u in [0,1/4]
all decisions are feasible, g=u and the supporting sine-power p=2
theorem proves exact global PF membership from
 ((5/4)*pi_hi/16)^2 < 1-2/pi_lo.

On this patch ||dF/dx1|| <= (5/4)*pi/2 and
||dF/du|| <=1+pi/2 via the original spherical equations.
The exact u-grid is interpreted by decision x4=u**10 (not rounded
or approximated). For max gaps dx,du the raw Euclidean covering is
 <= ((5/4)*pi_hi/2)*dx/2+(1+pi_hi/2)*du/2.

A mathematically sound proof for this RESTRICTED positive-g patch
and a synthetic 3x3 exact grid, NOT an accepted production GT or
continuous coverage of the entire positive-g component.
"""
from __future__ import annotations

from fractions import Fraction

from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz56_exact_same_ray import _encode
from .dtlz56_dtlz6_radial_enclosure import (
    certify_dtlz6_radial_interval_pf,replay_dtlz6_radial_interval_pf,
)

PROOF_ID="DTLZ6-M4K1-UNIFORM-POSITIVE-G-PATCH-COVER-V1"
MAX_X1=Fraction(1,8)
MAX_U=Fraction(1,4)


def _exact_axis(axis,*,minimum,maximum):
    if not isinstance(axis,(tuple,list)) or len(axis)<2:
        raise ValueError("At least two ordered grid values required")
    try:
        values=tuple(Fraction.from_float(float(v)) for v in axis)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("Expected finite exact binary grid entries") from exc
    if (values[0]!=minimum or values[-1]!=maximum or any(
        not minimum<=a<b<=maximum for a,b in zip(values,values[1:])
    )):
        raise ValueError("Grid must include exact endpoints and increase strictly")
    return values


def certify_dtlz6_positive_g_patch_cover(
    *,x1_grid=(0.,1/16,1/8),u_grid=(0.,1/8,1/4),
    pi_terms=16,
):
    if type(pi_terms) is not int or not 4<=pi_terms<=32:
        raise ValueError("Invalid exact pi enclosure term count")
    xgrid=_exact_axis(x1_grid,minimum=Fraction(0),maximum=MAX_X1)
    ugrid=_exact_axis(u_grid,minimum=Fraction(0),maximum=MAX_U)
    pi_lo,pi_hi=_pi_rational_bounds(pi_terms)
    if not 3<pi_lo<pi_hi<4:
        raise ArithmeticError("Invalid Machin pi enclosure")
    rhi=Fraction(5,4)
    margin=1-Fraction(2,1)/pi_lo-(rhi*pi_hi/16)**2
    if margin<=0:
        raise AssertionError("Uniform exact positive-g proof condition failed")
    dx=max(b-a for a,b in zip(xgrid,xgrid[1:]))
    du=max(b-a for a,b in zip(ugrid,ugrid[1:]))
    upper=(rhi*pi_hi/2)*dx/2+(1+pi_hi/2)*du/2
    grid=[]
    for x in xgrid:
        for u in ugrid:
            tail=u**10
            decision=(x,Fraction(1),Fraction(1),tail)
            if u==0:
                basis="DTLZ56-UNIT-RADIUS"
            else:
                proof=certify_dtlz6_radial_interval_pf(
                    decision[:3],decision[3:],k=1,
                    root_bits=40,pi_terms=8)
                if not replay_dtlz6_radial_interval_pf(
                    decision[:3],decision[3:],proof.to_dict(),
                    k=1,root_bits=40,pi_terms=8):
                    raise AssertionError("Exact tenth-root support theorem did not replay")
                basis=proof.proof_id
            grid.append({
                "decision": [_encode(v) for v in decision],
                "exact_radial_u":_encode(u),
                "global_pf_membership_proof":basis,
            })
    return {
        "proof_id":PROOF_ID,
        "problem":"DTLZ6","M":4,"K":1,
        "conclusion":"rigorously_bounded_continuous_pf_patch",
        "patch_x1":[_encode(Fraction(0)),_encode(MAX_X1)],
        "patch_u":[_encode(Fraction(0)),_encode(MAX_U)],
        "tail_parameterization":"x4=u^10, exact real",
        "support_x2_x3":[_encode(Fraction(1)),_encode(Fraction(1))],
        "x1_grid":[_encode(v) for v in xgrid],
        "u_grid":[_encode(v) for v in ugrid],
        "grid_decision_images":grid,
        "uniform_pf_strict_margin":_encode(margin),
        "grid_max_x1_gap":_encode(dx),
        "grid_max_u_gap":_encode(du),
        "raw_euclidean_continuous_patch_cover_upper":_encode(upper),
        "pi_terms":pi_terms,
        "continuous_subset_only_not_whole_positive_g_component":True,
        "reference_is_synthetic_exact_grid_not_production_gt":True,
        "whole_gt_coverage":"unverified",
        "indicator_fitness":"unverified",
        "scientific_certification":"unverified",
    }


def replay_dtlz6_positive_g_patch_cover(
    record,*,x1_grid=(0.,1/16,1/8),u_grid=(0.,1/8,1/4),
    pi_terms=16,
):
    try:
        expected=certify_dtlz6_positive_g_patch_cover(
            x1_grid=x1_grid,u_grid=u_grid,pi_terms=pi_terms)
        return (
            record==expected and
            expected["conclusion"]=="rigorously_bounded_continuous_pf_patch"
        )
    except (ValueError,TypeError,ArithmeticError,AssertionError,
            ZeroDivisionError,OverflowError):
        return False
