# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Certified continuous positive-g *patch* and grid coverage for original DTLZ5.

This is a restricted 2-D subset of the ORIGINAL M=4,K=1 PF, not the
entire positive-g component. Exact dyadic parameter rectangle:

  x1 in [0,1/8], x2=x3=1, tail t in [1/2,3/4],
  g=(t-1/2)^2 in [0,1/16].

Using original angular equations, the global sine-power support theorem
(p=2) proves that EVERY point in the patch is globally Pareto optimal:
  c²=(1+g)² sin²(pi*x1/2)
      <= ((17/16)*pi_hi/16)² < 1-2/pi_lo.
This strict comparison is EXACT rational, valid uniformly in the patch.

For F=r U(theta1,theta2,theta3), the spherical coordinate derivatives
satisfy ||dU/dtheta_j||<=1. Along the patch, r<=17/16,
|dtheta1/dx1|=pi/2 and dtheta2/dg=dtheta3/dg=pi/(4r²).
Thus ||dF/dx1||<=17*pi/32 and
||dF/dt||<= |dg/dt|*(1+pi/(2r)) <= 1+pi/2,
where |dg/dt|=2(t-1/2)<=1/2 (bound deliberately loosened to 1).
For grid maximum x1 and t spacings dx,dt, nearest grid coordinates
are within dx/2,dt/2: continuous raw Euclidean patch-cover <=
 (17*pi_hi/32)*(dx/2)+(1+pi_hi/2)*(dt/2).

This is a proved bound for the *synthetic grid reference* on the
specified patch, NOT source-selected GT coverage, exact stored F,
an entire positive-g component, or whole-GT indicator certification.
"""
from __future__ import annotations

from fractions import Fraction
from .dtlz56_exact_same_ray import _encode
from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz56_tight_sine_global_pf import (
    certify_tight_sine_global_pf, replay_tight_sine_global_pf,
)

PROOF_ID="DTLZ5-M4K1-UNIFORM-POSITIVE-G-PATCH-COVER-V1"
MAX_X1=Fraction(1,8)
TAIL_LO=Fraction(1,2)
TAIL_HI=Fraction(3,4)


def _exact_axis(values, *, low, high):
    """Strict increasing dyadic rational grid with both endpoints."""
    if not isinstance(values,(list,tuple)):
        raise ValueError("Expected ordered list/tuple of grid values")
    if len(values)<2:
        raise ValueError("At least two positions per axis")
    try:
        axis=tuple(Fraction.from_float(float(x)) for x in values)
    except (TypeError,ValueError,OverflowError) as ex:
        raise ValueError("Expected finite numeric axis entries") from ex
    if axis[0]!=low or axis[-1]!=high or any(
        not low<=a<b<=high for a,b in zip(axis,axis[1:])
    ):
        raise ValueError("Expected strictly increasing grid with exact endpoints")
    return axis


def certify_positive_g_patch_cover(
    *, x1_grid=(0.,1/16,1/8), tail_grid=(.5,.625,.75),
    pi_terms=16,
):
    """Exact uniform membership and continuous covering proof; no production."""
    if type(pi_terms) is not int or not 4<=pi_terms<=32:
        raise ValueError("pi_terms must be integer in [4,32]")
    xx=_exact_axis(x1_grid,low=Fraction(0),high=MAX_X1)
    tt=_exact_axis(tail_grid,low=TAIL_LO,high=TAIL_HI)
    pi_lo,pi_hi=_pi_rational_bounds(pi_terms)
    if not 3<pi_lo<pi_hi<4:
        raise ArithmeticError("Invalid exact Machin rational pi bounds")
    r_upper=Fraction(17,16)
    # sin(theta)<=theta, theta<=pi*x1/2 <=pi/16.
    c_sq_upper=(r_upper*pi_hi/16)**2
    threshold=1-Fraction(2,1)/pi_lo
    proof_margin=threshold-c_sq_upper
    if proof_margin<=0:
        raise AssertionError("Uniform positive-g membership strict proof failed")

    dx=max(b-a for a,b in zip(xx,xx[1:]))
    dt=max(b-a for a,b in zip(tt,tt[1:]))
    lx=r_upper*pi_hi/2
    # A deliberately conservative Lipschitz bound on the entire patch.
    lt=1+pi_hi/2
    covering_upper=lx*dx/2+lt*dt/2

    # Independently replay each *positive-g* dyadic grid member's existing
    # exact supporting sine theorem; g=0 uses unit-radius theorem.
    grid=[]
    for x1 in xx:
        for t in tt:
            g=(t-TAIL_LO)**2
            decision=(x1,Fraction(1),Fraction(1),t)
            if g==0:
                basis="DTLZ56-UNIT-RADIUS"
            else:
                certificate=certify_tight_sine_global_pf(
                    decision[:3],g=g,problem="DTLZ5",k=1)
                if not replay_tight_sine_global_pf(
                    decision[:3],certificate.to_dict(),
                    g=g,problem="DTLZ5",k=1):
                    raise AssertionError("Exact grid PF membership replay failed")
                basis=certificate.proof_id
            grid.append({
                "decision": [_encode(v) for v in decision],
                "original_g":_encode(g),
                "global_pf_membership_proof":basis,
            })
    return {
        "proof_id":PROOF_ID,
        "problem":"DTLZ5","M":4,"K":1,
        "conclusion":"rigorously_bounded_continuous_pf_patch",
        "patch_x1":[_encode(Fraction(0)),_encode(MAX_X1)],
        "patch_tail":[_encode(TAIL_LO),_encode(TAIL_HI)],
        "support_x2_x3":[_encode(Fraction(1)),_encode(Fraction(1))],
        "x1_grid":[_encode(v) for v in xx],
        "tail_grid":[_encode(v) for v in tt],
        "grid_decision_images":grid,
        "uniform_pf_strict_margin":_encode(proof_margin),
        "grid_max_x1_gap":_encode(dx),
        "grid_max_tail_gap":_encode(dt),
        "raw_euclidean_continuous_patch_cover_upper":_encode(covering_upper),
        "pi_terms":pi_terms,
        "continuous_subset_only_not_whole_positive_g_component":True,
        "reference_is_synthetic_exact_grid_not_production_gt":True,
        "whole_gt_coverage":"unverified",
        "indicator_fitness":"unverified",
        "scientific_certification":"unverified",
    }


def replay_positive_g_patch_cover(record, *, x1_grid=(0.,1/16,1/8),
                                  tail_grid=(.5,.625,.75),pi_terms=16):
    """Recompute all exact rational bounds; reject corruption and abstention."""
    try:
        expected=certify_positive_g_patch_cover(
            x1_grid=x1_grid,tail_grid=tail_grid,pi_terms=pi_terms)
        return (
            record==expected and
            expected["conclusion"]=="rigorously_bounded_continuous_pf_patch"
        )
    except (ValueError,TypeError,ArithmeticError,AssertionError,
            ZeroDivisionError,OverflowError):
        return False
