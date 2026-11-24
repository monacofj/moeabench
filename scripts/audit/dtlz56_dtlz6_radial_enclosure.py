# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rigorous *irrational-g* positive-PF certificates for original DTLZ6.

A stored IEEE-754 decision is an exact dyadic rational declaration. Its
original DTLZ6 radial distance g=sum(x_tail**(1/10)) is often irrational.
Replacing g with the production evaluator's floating approximation
would make a supposedly exact PF theorem unsound. Instead, bound each
tenth root between dyadic rationals by EXACT integer comparisons, then
use the *upper* rational g bound in the already proven sufficient global
sine-power support theorem for M>=4.

For an inner-angle support pattern j with exponent p>=2, the candidate
at the exact real g has c=(1+g)*sin(pi*x1/2). If the exact rational
upper bound B=(1+g_upper)*pi_upper*x1/2 satisfies
    B*B < 1 - 4/(p*pi_lower),
then the actual c meets the strict global monotonicity theorem; the
exact-real original objective image is globally Pareto optimal.
No rationality assumption on g is needed. This is NOT a certification
of finite-precision stored F, sampling coverage, or ADR0059 acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .dtlz56_rational_interval import (
    _as_exact_fraction, _pi_rational_bounds,
)
from .dtlz56_generalized_positive_g_pf import _supporting_objective
from .dtlz56_exact_same_ray import _encode


def rational_tenth_root_enclosure(value, *, bits=40):
    """Exact outward dyadic enclosure of the positive real tenth root.

    This uses integer power comparisons only: no floating-point roots,
    transcendental evaluation, or approximation hidden as proof.
    """
    if type(bits) is not int or not 4 <= bits <= 80:
        raise ValueError("bits must be an integer from 4 to 80")
    x = _as_exact_fraction(value, name="tail")
    if not 0 <= x <= 1:
        raise ValueError("Original DTLZ6 tail must belong to [0,1]")
    if x == 0 or x == 1:
        return (x, x)
    scale = 1 << bits
    target = x.numerator * scale**10
    denominator = x.denominator
    left, right = 0, scale
    while left < right:
        middle = (left + right + 1) // 2
        if middle**10 * denominator <= target:
            left = middle
        else:
            right = middle - 1
    lower = Fraction(left, scale)
    if left**10 * denominator == target:
        return (lower, lower)
    return (lower, Fraction(left + 1, scale))


@dataclass(frozen=True)
class DTLZ6RadialIntervalCertificate:
    conclusion: str
    problem: str
    dimension: int
    tail_dimension: int
    positions: tuple[dict, ...]
    tail: tuple[dict, ...]
    g_lower: dict
    g_upper: dict
    root_bits: int
    pi_terms: int
    supporting_objective: int | None
    sine_power: int | None
    strict_margin: dict | None
    proof_id: str = "DTLZ6-IRRATIONAL-G-UPPER-RADIAL-SUPPORT-V1"
    basis: str = (
        "Exact integer tenth-power root enclosures and rational Machin pi "
        "bounds imply global positive-g support monotonicity"
    )

    def to_dict(self):
        return {
            "conclusion": self.conclusion,
            "problem": self.problem,
            "dimension": self.dimension,
            "tail_dimension": self.tail_dimension,
            "positions": list(self.positions),
            "tail": list(self.tail),
            "g_lower": self.g_lower,
            "g_upper": self.g_upper,
            "root_bits": self.root_bits,
            "pi_terms": self.pi_terms,
            "supporting_objective": self.supporting_objective,
            "sine_power": self.sine_power,
            "strict_margin": self.strict_margin,
            "proof_id": self.proof_id,
            "basis": self.basis,
            "scientific_certification": "unverified",
        }


def certify_dtlz6_radial_interval_pf(
    positions, tail, *, k, root_bits=40, pi_terms=8,
):
    """Exact conditional *global* proof with possibly irrational g.

    No claim is made if the sufficient support condition is not proved.
    Input float values are taken as their EXACT binary rationals.
    """
    if type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ6 integer K>=1")
    if type(root_bits) is not int or not 4 <= root_bits <= 80:
        raise ValueError("root_bits must be integer in [4,80]")
    if type(pi_terms) is not int or not 2 <= pi_terms <= 32:
        raise ValueError("pi_terms must be integer in [2,32]")
    try:
        u = tuple(_as_exact_fraction(x, name="position") for x in positions)
        tail_values = tuple(_as_exact_fraction(x, name="tail") for x in tail)
    except TypeError as ex:
        raise ValueError("Expected original position/tail vectors") from ex
    if len(u) < 3 or any(not 0 <= x <= 1 for x in u):
        raise ValueError("Expected M>=4 and original positions in [0,1]")
    if len(tail_values) != k or any(not 0 <= x <= 1 for x in tail_values):
        raise ValueError("Expected K original tail variables in [0,1]")

    bounds = tuple(
        rational_tenth_root_enclosure(x, bits=root_bits)
        for x in tail_values
    )
    lo = sum((x[0] for x in bounds), Fraction(0))
    hi = min(Fraction(k), sum((x[1] for x in bounds), Fraction(0)))
    # g>0 if any original tail coordinate is strictly positive,
    # even if the finite dyadic root LOWER bound happens to be zero.
    positive = any(x > 0 for x in tail_values)
    j, p = _supporting_objective((Fraction(0),) + u[1:])
    margin = None
    certified = False
    if positive and j is not None:
        pi_lo, pi_hi = _pi_rational_bounds(pi_terms)
        b_squared = ((1 + hi) * pi_hi * u[0] / 2)**2
        threshold = 1 - Fraction(4, p * pi_lo)
        margin = threshold - b_squared
        certified = margin > 0

    return DTLZ6RadialIntervalCertificate(
        "rigorously_global_pf_member" if certified else "unverified",
        "DTLZ6", len(u) + 1, k,
        tuple(_encode(v) for v in u), tuple(_encode(v) for v in tail_values),
        _encode(lo), _encode(hi), root_bits, pi_terms, j, p,
        _encode(margin) if margin is not None else None,
    )


def replay_dtlz6_radial_interval_pf(
    positions, tail, record, *, k, root_bits=40, pi_terms=8,
):
    """Recompute the full exact dyadic root proof and disallow tampering."""
    try:
        supplied = record.to_dict() if isinstance(
            record, DTLZ6RadialIntervalCertificate
        ) else record
        expected = certify_dtlz6_radial_interval_pf(
            positions, tail, k=k, root_bits=root_bits, pi_terms=pi_terms,
        ).to_dict()
        return (
            supplied == expected
            and expected["conclusion"] == "rigorously_global_pf_member"
        )
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return False
