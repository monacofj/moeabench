# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded deterministic search for rigorously certified DTLZ5/6 tangents.

This module proposes a *finite* set of simple rational one-sided tangent
directions. Every accepted direction must pass the exact-rational derivative
enclosure theorem in dtlz56_rational_tangent.py. A failed search yields
UNVERIFIED, never a proof of stationarity or Pareto membership.

No linear solver, floating-point trigonometry, production MOP evaluator,
GT generator, or unbounded search is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .dtlz56_rational_interval import _as_exact_fraction
from .dtlz56_rational_tangent import (
    certify_rational_tangent_descent,
    replay_rational_tangent_certificate,
)
from .dtlz56_exact_same_ray import _encode


@dataclass(frozen=True)
class RationalTangentSearch:
    outcome: str
    candidates_examined: int
    candidate_limit: int
    velocity: tuple[dict, ...] | None
    derivative_certificate: dict | None
    basis: str = (
        "Finite deterministic rational tangent candidates; every accepted "
        "direction is checked with exact Machin/Taylor derivative enclosures"
    )

    def to_dict(self):
        return {
            "outcome": self.outcome,
            "candidates_examined": self.candidates_examined,
            "candidate_limit": self.candidate_limit,
            "velocity": list(self.velocity) if self.velocity is not None else None,
            "derivative_certificate": self.derivative_certificate,
            "basis": self.basis,
            "scientific_certification": "unverified",
        }


def _directions(dimension):
    """Generate low-complexity rational directions without duplicates."""
    zero = (Fraction(0),) * dimension
    first_values = (
        Fraction(-2, 5), Fraction(0), Fraction(-1, 4),
        Fraction(1, 4), Fraction(2, 5), Fraction(-1, 2),
        Fraction(1, 2),
    )
    seen = set()
    for first in first_values:
        proposal = (first,) + zero[1:]
        if proposal not in seen:
            seen.add(proposal)
            yield proposal
    for coord in range(1, dimension):
        for value in (Fraction(-1, 2), Fraction(1, 2),
                      Fraction(-1), Fraction(1)):
            for first in (Fraction(0), Fraction(-2, 5),
                          Fraction(2, 5)):
                candidate = list(zero)
                candidate[0], candidate[coord] = first, value
                proposal = tuple(candidate)
                if proposal not in seen:
                    seen.add(proposal)
                    yield proposal


def _one_sided_feasible(u, v):
    return all(
        not (x == 0 and speed > 0 or x == 1 and speed < 0)
        for x, speed in zip(u, v)
    )


def search_rational_tangent_witness(
    positions, *, g, problem, k, max_candidates=16,
    pi_terms=6, trig_pairs=2,
):
    """Find one rigorous local dominator, or return an unresolved outcome.

    max_candidates bounds only evaluated feasible candidate directions.
    The search does not conclude anything when candidates are exhausted.
    """
    if type(max_candidates) is not int or not 1 <= max_candidates <= 128:
        raise ValueError("max_candidates must be integer in [1,128]")
    try:
        positions = tuple(_as_exact_fraction(x, name="position")
                          for x in positions)
    except TypeError as exc:
        raise ValueError("Expected position vector") from exc
    # This initial call validates original problem, g and positions, as
    # well as the selected precision budget, even for early success.
    certify_rational_tangent_descent(
        positions, g=g, velocity=(Fraction(0),)*len(positions),
        problem=problem, k=k, pi_terms=pi_terms, trig_pairs=trig_pairs,
    )
    count = 0
    for direction in _directions(len(positions)):
        if not _one_sided_feasible(positions, direction):
            continue
        if count == max_candidates:
            break
        certificate = certify_rational_tangent_descent(
            positions, g=g, velocity=direction, problem=problem, k=k,
            pi_terms=pi_terms, trig_pairs=trig_pairs,
        )
        count += 1
        if certificate.conclusion == "rigorous_local_dominator":
            return RationalTangentSearch(
                "rigorous_local_dominator_found", count, max_candidates,
                tuple(_encode(v) for v in direction), certificate.to_dict(),
            )
    return RationalTangentSearch(
        "unverified", count, max_candidates, None, None,
    )


def replay_rational_tangent_search(
    positions, record, *, g, problem, k,
    pi_terms=6, trig_pairs=2,
):
    """Re-evaluate searched proposal order and its independently checked proof."""
    try:
        supplied = record.to_dict() if isinstance(record, RationalTangentSearch) else record
        limit = supplied["candidate_limit"]
        repeated = search_rational_tangent_witness(
            positions, g=g, problem=problem, k=k,
            max_candidates=limit, pi_terms=pi_terms,
            trig_pairs=trig_pairs,
        ).to_dict()
        if supplied != repeated or repeated["outcome"] != "rigorous_local_dominator_found":
            return False
        velocity = tuple(
            Fraction(int(x["numerator_hex"], 16), int(x["denominator_hex"], 16))
            for x in repeated["velocity"]
        )
        return replay_rational_tangent_certificate(
            positions, repeated["derivative_certificate"],
            g=g, velocity=velocity, problem=problem, k=k,
            pi_terms=pi_terms, trig_pairs=trig_pairs,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError, ArithmeticError):
        return False
