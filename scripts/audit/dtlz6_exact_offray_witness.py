# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact outward rational off-ray DTLZ6 M4 K1 strict domination witness.

Numerical optimization is permitted ONLY to propose feasible witnesses;
all claimed dominance is independently re-evaluated with exact IEEE-754
binary decision coordinates, rigorous integer tenth-root enclosures, and
outward Machin/Taylor trigonometric original-equation intervals.
A strict positive LOWER separation in ALL four objectives proves
dominance. Any nonpositive separation or incomplete input ABSTAINS.
The code does not alter source sampling, accepted ADR or regression tests.
"""
from __future__ import annotations

from fractions import Fraction

from .dtlz5_patch_actual_source_cover import _exact_decisions
from .dtlz6_patch_actual_source_cover import _outward_original_objectives
from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz56_exact_same_ray import _encode

SCHEMA="DTLZ6-M4K1-EXACT-INTERVAL-STRICT-OFFRAY-WITNESS-V1"


def certify_dtlz6_offray_dominator(source,witness,*,root_bits=64,pi_terms=16):
    if type(root_bits) is not int or not 8<=root_bits<=80:
        raise ValueError("Unsupported integer tenth-root enclosure")
    if type(pi_terms) is not int or not 4<=pi_terms<=32:
        raise ValueError("Unsupported exact pi enclosure")
    x,z=_exact_decisions((source,witness))
    pi=_pi_rational_bounds(pi_terms)
    origin=_outward_original_objectives(x,pi,root_bits=root_bits)
    alternative=_outward_original_objectives(z,pi,root_bits=root_bits)
    separation=tuple(origin[i][0]-alternative[i][1] for i in range(4))
    positive=all(v>0 for v in separation)
    return {
        "schema":SCHEMA,"problem":"DTLZ6","M":4,"K":1,
        "conclusion":"rigorously_strictly_dominated" if positive else "unverified",
        "source":[_encode(v) for v in x],
        "witness":[_encode(v) for v in z],
        "pi_terms":pi_terms,"root_bits":root_bits,
        "strict_original_objective_separation_lower":[
            _encode(v) for v in separation
        ],
        "original_source_outward_intervals":[
            [_encode(lo),_encode(hi)] for lo,hi in origin
        ],
        "witness_original_outward_intervals":[
            [_encode(lo),_encode(hi)] for lo,hi in alternative
        ],
        "finite_archived_F_exactness":"unverified",
        "full_gt_coverage":"unverified",
        "scientific_certification":"unverified",
    }


def replay_dtlz6_offray_dominator(
    source,witness,record,*,root_bits=64,pi_terms=16
):
    try:
        r=certify_dtlz6_offray_dominator(
            source,witness,root_bits=root_bits,pi_terms=pi_terms)
        return r==record and r["conclusion"]=="rigorously_strictly_dominated"
    except (ValueError,TypeError,ArithmeticError,
            AssertionError,ZeroDivisionError,OverflowError):
        return False
