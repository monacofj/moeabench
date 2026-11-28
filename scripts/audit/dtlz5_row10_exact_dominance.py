# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent exact-rational strict dominance proof for original DTLZ5 M=4,K=1.

For every source and witness input, interpret IEEE-754 coordinates as their
EXACT dyadic rationals. Evaluate the original analytical hyperspherical
objectives using outward rational Machin pi bounds and one-sided Taylor
sine/cosine bounds; no production evaluator, sampler, solver or float
trigonometry participates. A strict positive interval separation in ALL
four objectives proves exact-real strict dominance. Otherwise abstain.

This is decision-image proof, NOT exact stored-float F identity, GT coverage,
numerical platform/replay certification or ADR 0059 whole-GT acceptance.
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial

from .dtlz56_rational_interval import _as_exact_fraction, _pi_rational_bounds
from .dtlz56_exact_same_ray import _encode

PROOF_ID = "DTLZ5-M4K1-EXACT-TAYLOR-STRICT-DOMINATOR-V1"


def _sine_bounds(x: Fraction):
    """Alternating rational bounds on sin(x), 0<=x<=pi/2<2."""
    if not 0 <= x < 2:
        raise ValueError("Sine Taylor range exceeded")
    upper = sum(
        (-1)**j * x**(2*j+1) / factorial(2*j+1)
        for j in range(9)
    )  # degree 17, positive last term
    return upper - x**19 / factorial(19), upper


def _cosine_bounds(x: Fraction):
    """Alternating rational bounds on cos(x), 0<=x<=pi/2<2."""
    if not 0 <= x < 2:
        raise ValueError("Cosine Taylor range exceeded")
    upper = sum(
        (-1)**j * x**(2*j) / factorial(2*j)
        for j in range(11)
    )  # degree 20, positive last term
    return upper - x**22 / factorial(22), upper


def _trig_bounds(pi_multiple, pi_bounds, *, sine):
    lo, hi = (pi_bounds[0]*pi_multiple, pi_bounds[1]*pi_multiple)
    if not 0 <= lo <= hi < 2:
        raise ValueError("Original angular range exceeded")
    if sine:
        # Sine is nondecreasing on [0, pi/2].
        return _sine_bounds(lo)[0], _sine_bounds(hi)[1]
    # Cosine is nonincreasing on [0, pi/2].
    return _cosine_bounds(hi)[0], _cosine_bounds(lo)[1]


def _original_objective_intervals(decision, pi_bounds):
    """Outward rational F intervals, original DTLZ5 M4 K1 equations."""
    x1, x2, x3, tail = decision
    g = (tail - Fraction(1, 2))**2
    coeffs = (
        x1 / 2,
        (1+2*g*x2) / (4*(1+g)),
        (1+2*g*x3) / (4*(1+g)),
    )
    if any(not 0 <= z <= Fraction(1, 2) for z in coeffs):
        raise ValueError("Original DTLZ5 angle out of range")
    cosines = [_trig_bounds(c, pi_bounds, sine=False) for c in coeffs]
    sines = [_trig_bounds(c, pi_bounds, sine=True) for c in coeffs]
    pieces = (
        (cosines[0], cosines[1], cosines[2]),
        (cosines[0], cosines[1], sines[2]),
        (cosines[0], sines[1]),
        (sines[0],),
    )
    output = []
    for factors in pieces:
        lower, upper = 1+g, 1+g
        for a, b in factors:
            if not 0 <= a <= b:
                raise ArithmeticError("Nonnegative outward trigonometric bounds required")
            lower *= a
            upper *= b
        output.append((lower, upper))
    return output


def certify_strict_dtlz5_m4_dominator(
    source, witness, *, pi_terms=16,
):
    """Return a strict mathematical dominance proof, or abstain."""
    if type(pi_terms) is not int or not 4 <= pi_terms <= 32:
        raise ValueError("pi_terms must be integer in [4,32]")
    try:
        x = tuple(_as_exact_fraction(z, name="source") for z in source)
        z = tuple(_as_exact_fraction(t, name="witness") for t in witness)
    except TypeError as exc:
        raise ValueError("Expected four original decision coordinates") from exc
    if len(x) != 4 or len(z) != 4 or any(not 0 <= a <= 1 for a in x+z):
        raise ValueError("Expected original M=4 K=1 feasible decisions")
    source_g = (x[3]-Fraction(1, 2))**2
    witness_g = (z[3]-Fraction(1, 2))**2
    pi_bounds = _pi_rational_bounds(pi_terms)
    if not 3 < pi_bounds[0] < pi_bounds[1] < 4:
        raise ArithmeticError("Invalid Machin pi bounds")
    original = _original_objective_intervals(x, pi_bounds)
    alternative = _original_objective_intervals(z, pi_bounds)
    separations = tuple(original[i][0]-alternative[i][1] for i in range(4))
    proved = all(delta > 0 for delta in separations)
    return {
        "proof_id": PROOF_ID,
        "conclusion": ("rigorously_strictly_dominated"
                       if proved else "unverified"),
        "problem": "DTLZ5",
        "M": 4, "K": 1, "pi_terms": pi_terms,
        "source": [_encode(v) for v in x],
        "witness": [_encode(v) for v in z],
        "source_g": _encode(source_g),
        "witness_g": _encode(witness_g),
        "strict_objective_margins": [_encode(v) for v in separations],
        "exact_decision_image_only": True,
        "scientific_certification": "unverified",
    }


def replay_strict_dtlz5_m4_dominator(
    source, witness, record, *, pi_terms=16,
):
    """Recompute every rational bound; reject tampering or nonproofs."""
    try:
        expected = certify_strict_dtlz5_m4_dominator(
            source, witness, pi_terms=pi_terms,
        )
        return (record == expected and
                expected["conclusion"] == "rigorously_strictly_dominated")
    except (TypeError, ValueError, ArithmeticError, ZeroDivisionError):
        return False


# Exact IEEE-754 values from the post-PR #210 original M4 K1 final-selected
# n16 DTLZ5 source row 10. The witness uses rational binary displacements.
SOURCE_ROW_10 = (
    0.1671198950208142, 0.999999658561122, 1.0, 0.6805481014263987,
)


def archived_row10_witness():
    """Construct fixed dyadic witness independently of any optimizer."""
    original = tuple(Fraction.from_float(t) for t in SOURCE_ROW_10)
    return (
        original[0]-Fraction(1, 2**31),
        original[1]+Fraction(1, 2**23),
        original[2],
        original[3]-Fraction(1, 2**26),
    )
