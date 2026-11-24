# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rigorous first-order different-ray dominance for original DTLZ5/6.

For rational feasible (u,g), g>0 and rational feasible one-sided
velocity v, enclose every dF_i(g,u+(g-g0)*v)/dg at g0 using exact
Fraction arithmetic, certified Machin bounds for pi and alternating
Taylor bounds for sine/cosine on [0,pi/2]. If ALL derivative lower
bounds are strictly positive, the continuity theorem gives a feasible
dominator at some arbitrarily close g'<g, including angular boundaries.

This is an *existence proof*, not a numerically reconstructed decision
witness. Original-image alignment and scientific acceptance of a stored
GT row are separate ADR 0059 obligations. Never certify nondominance
from a nonpositive or unverified derivative.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial

from .dtlz56_rational_interval import (
    _as_exact_fraction, _pi_rational_bounds
)
from .dtlz56_exact_same_ray import _encode


def _add(a, b):
    return (a[0]+b[0], a[1]+b[1])


def _neg(a):
    return (-a[1], -a[0])


def _multiply(a, b):
    products = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
    return (min(products), max(products))


def _scale(a, q):
    return _multiply(a, (q, q))


def _product(parts):
    out = (Fraction(1), Fraction(1))
    for part in parts:
        out = _multiply(out, part)
    return out


def _sin_taylor(x, n):
    return sum(
        (-1)**j*x**(2*j+1)/factorial(2*j+1)
        for j in range(n)
    )


def _cos_taylor(x, n):
    return sum(
        (-1)**j*x**(2*j)/factorial(2*j)
        for j in range(n)
    )


def _trigonometric_bounds(angle, pairs):
    """Outward exact rational bounds on sine/cosine for 0<=angle<=pi/2.

    Sine is increasing, cosine decreasing. Even-length alternating
    sine sums bound below, odd-length bound above; for cosine,
    even-length sums bound below, odd-length bound above.
    """
    a, b = angle
    assert 0 <= a <= b < 2
    sin_interval = (
        max(Fraction(0), _sin_taylor(a, 2*pairs)),
        min(Fraction(1), _sin_taylor(b, 2*pairs+1)),
    )
    cos_interval = (
        max(Fraction(0), _cos_taylor(b, 2*pairs)),
        min(Fraction(1), _cos_taylor(a, 2*pairs+1)),
    )
    if sin_interval[0] > sin_interval[1] or cos_interval[0] > cos_interval[1]:
        raise ArithmeticError("Trigonometric bounds not ordered")
    return sin_interval, cos_interval


@dataclass(frozen=True)
class RationalTangentCertificate:
    conclusion: str
    original_g: dict
    velocity: tuple[dict, ...]
    derivative_intervals: tuple[dict, ...]
    basis: str = (
        "Original hyperspherical product derivatives, exact-rational "
        "Machin/Taylor interval bounds and one-sided feasible tangent; "
        "strict positive lower derivatives prove local dominance"
    )

    def to_dict(self):
        return {
            "conclusion": self.conclusion,
            "original_g": self.original_g,
            "velocity": list(self.velocity),
            "derivative_intervals": list(self.derivative_intervals),
            "basis": self.basis,
            "scientific_certification": "unverified",
        }


def certify_rational_tangent_descent(
    positions, *, g, velocity, problem, k, pi_terms=6, trig_pairs=2,
):
    """Proof of some strict local g-reducing dominator, or abstention."""
    if problem not in ("DTLZ5", "DTLZ6") or type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ5/6 and K>=1")
    if type(pi_terms) is not int or not 2 <= pi_terms <= 24:
        raise ValueError("pi_terms must be integer in [2,24]")
    if type(trig_pairs) is not int or not 1 <= trig_pairs <= 8:
        raise ValueError("trig_pairs must be integer in [1,8]")
    try:
        u = [_as_exact_fraction(x, name="position") for x in positions]
        v = [_as_exact_fraction(x, name="velocity") for x in velocity]
    except TypeError as ex:
        raise ValueError("Expected position/velocity vectors") from ex
    if not u or len(v) != len(u) or any(x<0 or x>1 for x in u):
        raise ValueError("Expected aligned M>=2 positions within [0,1]")
    gy = _as_exact_fraction(g, name="g")
    gmax = Fraction(k,4) if problem == "DTLZ5" else Fraction(k)
    if not 0 < gy <= gmax:
        raise ValueError("Expected feasible positive g")
    for x, speed in zip(u, v):
        # Path is u(g') = u + (g'-g)*v, g'<g.
        if (x == 0 and speed > 0) or (x == 1 and speed < 0):
            raise ValueError("Velocity points outside one-sided feasible box")

    pi_lo, pi_hi = _pi_rational_bounds(pi_terms)
    pi_interval = (pi_lo, pi_hi)
    theta = []
    slopes = []
    for i, pos in enumerate(u):
        if i == 0:
            theta.append(_scale(pi_interval, pos/2))
            slopes.append(_scale(pi_interval, v[i]/2))
        else:
            theta.append(_scale(
                pi_interval, (1+2*gy*pos)/(4*(1+gy))))
            coefficient = ((2*pos-1)/(4*(1+gy)**2)
                           + gy*v[i]/(2*(1+gy)))
            slopes.append(_scale(pi_interval, coefficient))
    trig = [_trigonometric_bounds(angle, trig_pairs) for angle in theta]
    m = len(u)+1
    derivatives = []
    for j in range(m):
        factors = []
        factor_derivatives = []
        for idx in range(m-1):
            if idx < m-j-1:
                sine, cosine = trig[idx]
                factors.append(cosine)
                factor_derivatives.append(
                    _multiply(_neg(sine), slopes[idx]))
            elif j > 0 and idx == m-j-1:
                sine, cosine = trig[idx]
                factors.append(sine)
                factor_derivatives.append(
                    _multiply(cosine, slopes[idx]))
            else:
                factors.append((Fraction(1), Fraction(1)))
                factor_derivatives.append((Fraction(0), Fraction(0)))
        product = _product(factors)
        derivative = product
        for idx, diff in enumerate(factor_derivatives):
            if diff == (0, 0):
                continue
            terms = [diff] + [
                factors[p] for p in range(m-1) if p != idx
            ]
            derivative = _add(
                derivative, _scale(_product(terms), 1+gy)
            )
        derivatives.append(derivative)

    return RationalTangentCertificate(
        "rigorous_local_dominator"
        if all(lower > 0 for lower, _ in derivatives)
        else "unverified",
        _encode(gy), tuple(_encode(x) for x in v),
        tuple({
            "lower": _encode(low), "upper": _encode(high)
        } for low, high in derivatives),
    )


def replay_rational_tangent_certificate(
    positions, record, *, g, velocity, problem, k,
    pi_terms=6, trig_pairs=2,
):
    """Independently recompute the exact interval certificate and compare."""
    try:
        original = record.to_dict() if isinstance(
            record, RationalTangentCertificate
        ) else record
        recalculated = certify_rational_tangent_descent(
            positions, g=g, velocity=velocity, problem=problem, k=k,
            pi_terms=pi_terms, trig_pairs=trig_pairs,
        ).to_dict()
        return (original == recalculated
                and original["conclusion"] == "rigorous_local_dominator")
    except (ValueError, TypeError, KeyError, ZeroDivisionError,
            ArithmeticError):
        return False
