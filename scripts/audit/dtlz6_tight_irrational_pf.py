# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Research-only tighter exact irrational-g PF support for original DTLZ6.

The existing DTLZ6 interval proof uses sin(theta)<=theta.
This independent opt-in certificate replaces that loose upper by
sin(theta)<=theta-theta^3/6+theta^5/120 (for 0<=theta<2),
using the exact rational UPPER of the ORIGINAL DTLZ6 real g
obtained by integer tenth-root outward enclosures of its tail.
A positive strict rational margin proves global PF membership
on the established supporting angular boundary patterns.
Abstains if the required inequality cannot be proved.
The same source is not allowed to upgrade the approved GT gate.
"""
from __future__ import annotations
from fractions import Fraction
from .dtlz56_dtlz6_radial_enclosure import rational_tenth_root_enclosure
from .dtlz56_rational_interval import _pi_rational_bounds,_as_exact_fraction
from .dtlz56_generalized_positive_g_pf import _supporting_objective
from .dtlz56_tight_sine_global_pf import _taylor_sine_upper
from .dtlz56_exact_same_ray import _encode

SCHEMA="DTLZ6-IRRATIONAL-RADIAL-TIGHT-SINE-SUPPORT-V1"


def certify_dtlz6_tight_radial_pf(positions,tail,*,k,root_bits=64,pi_terms=16):
    if type(k) is not int or k<1:
        raise ValueError("Expected original DTLZ6 K>=1")
    if type(root_bits) is not int or not 8<=root_bits<=80:
        raise ValueError("Expected exact integer tenth-root precision")
    if type(pi_terms) is not int or not 4<=pi_terms<=32:
        raise ValueError("Expected rigorous Machin pi precision")
    try:
        u=tuple(_as_exact_fraction(x,name="position") for x in positions)
        tails=tuple(_as_exact_fraction(x,name="tail") for x in tail)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("Exact real original declarations required") from exc
    if len(u)<3 or len(tails)!=k or any(not 0<=x<=1 for x in u+tails):
        raise ValueError("Expected feasible original decision x and tail")
    intervals=tuple(rational_tenth_root_enclosure(x,bits=root_bits)
                    for x in tails)
    gl=sum((x[0] for x in intervals),Fraction(0))
    gh=sum((x[1] for x in intervals),Fraction(0))
    if not 0<=gl<=gh<=k:
        raise ArithmeticError("Inconsistent outward root intervals")
    positive=any(x>0 for x in tails)
    j,p=_supporting_objective((Fraction(0),)+u[1:])
    pi_lo,pi_hi=_pi_rational_bounds(pi_terms)
    if not 3<pi_lo<pi_hi<4:
        raise ArithmeticError("No exact pi enclosure")
    theta=pi_hi*u[0]/2
    sine=_taylor_sine_upper(theta)
    # theta==0 is an exact zero; the Taylor upper is zero also.
    margin=None
    if j is not None and positive:
        bound=((1+gh)*sine)**2
        threshold=1-Fraction(4,p*pi_lo)
        margin=threshold-bound
    proved=margin is not None and margin>0
    return {
        "schema":SCHEMA,"problem":"DTLZ6","M":len(u)+1,"K":k,
        "positions":[_encode(v) for v in u],
        "tail":[_encode(v) for v in tails],
        "g_lower":_encode(gl),"g_upper":_encode(gh),
        "pi_terms":pi_terms,"root_bits":root_bits,
        "supporting_objective":j,"sine_power":p,
        "taylor_sine_theta_upper":_encode(sine),
        "strict_rational_margin":_encode(margin) if margin is not None else None,
        "conclusion":("rigorously_global_pf_member" if proved else "unverified"),
        "exact_original_decision_image_only":True,
        "finite_stored_F_alignment":"unverified",
        "whole_gt_coverage":"unverified",
        "scientific_certification":"unverified",
    }


def replay_dtlz6_tight_radial_pf(positions,tail,record,*,k,
                                 root_bits=64,pi_terms=16):
    try:
        expected=certify_dtlz6_tight_radial_pf(
            positions,tail,k=k,root_bits=root_bits,pi_terms=pi_terms)
        return record==expected and expected["conclusion"]=="rigorously_global_pf_member"
    except (ValueError,TypeError,KeyError,ArithmeticError,
            ZeroDivisionError,OverflowError):
        return False
