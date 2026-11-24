# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rigorous *interval exclusion* of original DTLZ5/6 fixed-g dominators.

The mathematical inequalities use exact Fraction arithmetic. Pi is enclosed
by an alternating rational Machin/atan series, and the relevant sine is
bounded below by an alternating rational Taylor polynomial. The lower
squared-energy recurrence excludes ALL g in a specified closed rational
interval only when a strict exact-rational inequality proves infeasibility.

Inputs passed as Python finite floats are interpreted as their EXACT
binary rational values. This is deliberately a standalone research oracle:
- it does not prove that the supplied objective candidate is itself feasible;
- it does not certify the full [0,g_y) interval or its open endpoint;
- its result is not an environment/provenance/GT-adequacy certificate;
- public GTAssessment statuses remain unchanged under ADR 0059.

No hardware floating-point trigonometry, optimizer, evaluator, PS/PF,
production scalarizer, BLAS or nonstandard interval library is invoked.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial, isfinite
from typing import Sequence


@dataclass(frozen=True)
class RationalIntervalCertificate:
    conclusion: str
    g_lower: str
    g_upper: str
    stopping_objective: int | None
    strict_margin: str | None
    pi_terms: int
    sine_pairs: int
    basis: str = (
        "Machin exact-rational pi enclosure, alternating sine lower bound, "
        "monotone squared-residual continuous-g recurrence"
    )

    def to_dict(self):
        return {
            "conclusion": self.conclusion,
            "g_lower": self.g_lower,
            "g_upper": self.g_upper,
            "stopping_objective": self.stopping_objective,
            "strict_margin": self.strict_margin,
            "pi_terms": self.pi_terms,
            "sine_pairs": self.sine_pairs,
            "basis": self.basis,
            "scientific_certification": "unverified",
        }


def _arctan_rational_bounds(denominator: int, terms: int):
    """Exact alternating-series bounds for atan(1/denominator)."""
    x = Fraction(1, denominator)
    series = sum(
        ((-1)**j) * x**(2*j+1) / (2*j+1)
        for j in range(terms)
    )
    next_term = x**(2*terms+1) / (2*terms+1)
    return ((series, series+next_term) if terms % 2 == 0
            else (series-next_term, series))


def _pi_rational_bounds(terms: int):
    """Machin identity: pi = 16 atan(1/5) - 4 atan(1/239)."""
    a_lo, a_hi = _arctan_rational_bounds(5, terms)
    b_lo, b_hi = _arctan_rational_bounds(239, terms)
    return 16*a_lo-4*b_hi, 16*a_hi-4*b_lo


def _sine_lower_bound(x: Fraction, pairs: int):
    """Alternating 2*pairs-term Taylor lower bound for 0<=x<=1."""
    if not (0 <= x <= 1):
        raise ValueError("sine bound requires rational 0<=x<=1")
    lower = sum(
        ((-1)**j) * x**(2*j+1) / factorial(2*j+1)
        for j in range(2*pairs)
    )
    return lower


def _as_exact_fraction(value, *, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + " must be a finite real number") from exc
    if not isfinite(number):
        raise ValueError(name + " must be finite")
    return Fraction.from_float(number)


def rigorously_exclude_g_interval(
    objectives: Sequence[float], *, g_lower, g_upper,
    problem: str, k: int, pi_terms: int = 16,
    sine_pairs: int = 8,
) -> RationalIntervalCertificate:
    """Prove absence of D(g;y) for every g in [g_lower,g_upper], or abstain.

    Strict positive rational separation is required for any exclusion.
    An unresolved result does not imply existence of a feasible dominator.
    Floats are interpreted as exact binary rationals; no unproven
    outward-rounded float comparison is used.
    """
    if problem not in ("DTLZ5", "DTLZ6") or type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ5/6 with integer K>=1")
    if (type(pi_terms) is not int or pi_terms < 2 or
            type(sine_pairs) is not int or sine_pairs < 1):
        raise ValueError("Expected positive precision terms")
    if pi_terms > 64 or sine_pairs > 64:
        raise ValueError("Precision terms exceed supported exact-rational budget")

    try:
        coordinates = list(objectives)
    except TypeError as exc:
        raise ValueError("Expected objective vector") from exc
    if len(coordinates) < 2:
        raise ValueError("Expected objective vector M>=2")
    y = [_as_exact_fraction(v, name="objective cap") for v in coordinates]
    if any(v < 0 for v in y):
        raise ValueError("Objective caps must be nonnegative")

    lo = _as_exact_fraction(g_lower, name="g_lower")
    hi = _as_exact_fraction(g_upper, name="g_upper")
    limit = Fraction(k, 4) if problem == "DTLZ5" else Fraction(k)
    if not (0 <= lo <= hi <= limit):
        raise ValueError("Expected 0<=g_lower<=g_upper<=original g_max")

    pi_lo, pi_hi = _pi_rational_bounds(pi_terms)
    # At every g in [lo,hi], the inner angle
    # a(g)=pi/(4(1+g)) is >= this rational lower bound.
    angle_lo = pi_lo / (4*(1+hi))
    sin_lo = _sine_lower_bound(angle_lo, sine_pairs)
    if not (0 < sin_lo < 1 and pi_lo < pi_hi):
        raise ArithmeticError("Failed to establish rigorous sine/pi bounds")
    beta_lo = sin_lo*sin_lo

    # r(g)^2 increases with g, so this exact-rational q0 is a
    # lower bound for the first squared residual at all g in [lo,hi].
    q = max(Fraction(0), (1+lo)**2-y[-1]**2)
    conclusion = "unresolved"
    violated = None
    separation = None
    for j in range(len(y)-2, 0, -1):
        margin = q*beta_lo-y[j]**2
        if margin > 0:
            conclusion = "rigorously_excluded"
            violated = j+1
            separation = margin
            break
        q = max(q*beta_lo, q-y[j]**2)
    else:
        margin = q-y[0]**2
        if margin > 0:
            conclusion = "rigorously_excluded"
            violated = 1
            separation = margin

    return RationalIntervalCertificate(
        conclusion, str(lo), str(hi), violated,
        str(separation) if separation is not None else None,
        pi_terms, sine_pairs,
    )
