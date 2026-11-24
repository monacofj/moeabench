# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact-rational dominance test against the complete original DTLZ5/6 g=0 curve.

For nonnegative objective caps y, the entire curve C(t) is weakly below y
iff y[j]^2 >= a[j]^2*(1-min(y[M-1],1)^2) for j<M-1.
Here all a[j]^2 are powers of 1/2, so exact rational arithmetic suffices:
there are no trigonometric evaluations or approximate square roots.

If ||y||^2>1 and the curve fits, its unit-norm point strictly dominates y.
This is a *conditional* mathematical dominance theorem: separately verify
that a claimed GT row y really belongs to the original feasible image.
An absence of a g=0 curve dominator does not imply global Pareto optimality
when M>=4. Input floats are interpreted as exact binary rationals.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .dtlz56_rational_interval import _as_exact_fraction
from .dtlz56_exact_same_ray import _encode


@dataclass(frozen=True)
class ExactCurveDominance:
    conclusion: str
    tested_objectives: int
    obstructing_objective: int | None
    signed_squared_margin: dict | None
    radius_squared: dict
    basis: str = (
        "Exact rational test of complete original DTLZ5/6 g=0 "
        "curve dominance via dyadic squared coordinate inequalities"
    )

    def to_dict(self):
        return {
            "conclusion": self.conclusion,
            "tested_objectives": self.tested_objectives,
            "obstructing_objective": self.obstructing_objective,
            "signed_squared_margin": self.signed_squared_margin,
            "radius_squared": self.radius_squared,
            "basis": self.basis,
            "scientific_certification": "unverified",
        }


def exact_g0_curve_dominance(objectives, *, problem, k):
    """Decide g=0 curve weak dominance exactly, with safe strict status.

    A strict dominance result is conditional on the row's independent
    original-feasibility assessment. Returned failures concern *only* the
    complete g=0 curve and never certify global positive-g efficiency.
    """
    if problem not in ("DTLZ5", "DTLZ6") or type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ5/6 with integer K>=1")
    try:
        values = list(objectives)
    except TypeError as exc:
        raise ValueError("Expected an objective vector") from exc
    if len(values) < 2:
        raise ValueError("Expected at least two objectives")
    y = [_as_exact_fraction(v, name="objective") for v in values]
    if any(v < 0 for v in y):
        raise ValueError("Objectives must be nonnegative")
    m = len(y)
    norm_sq = sum(v*v for v in y)
    last = min(Fraction(1), y[-1])
    residual = 1-last*last
    obstruction = None
    signed_margin = None
    for j in range(m-1):
        # a_1^2 = 2^{-(M-2)}, a_j^2 = 2^{-(M-j)} (j >= 2).
        exponent = (m-2 if j == 0 else m-j-1)
        lower_squared = residual / (2**exponent)
        signed_margin = y[j]*y[j]-lower_squared
        if signed_margin < 0:
            obstruction = j+1
            break
    if obstruction is not None:
        conclusion = "no_g0_curve_dominator"
    elif norm_sq > 1:
        conclusion = "exact_g0_curve_dominator"
    elif norm_sq == 1:
        conclusion = "weak_g0_curve_fit_unit_radius"
    else:
        # Such y is outside the original feasible image; it cannot be
        # dominated by a g=0 unit-radius point despite passing none.
        conclusion = "outside_unit_radius_feasible_image"
    return ExactCurveDominance(
        conclusion, m-1 if obstruction is None else obstruction,
        obstruction,
        _encode(signed_margin) if signed_margin is not None else None,
        _encode(norm_sq),
    )


def replay_exact_g0_curve_dominance(objectives, record, *, problem, k):
    """Recompute all exact rational comparisons; reject forged outcomes."""
    try:
        provided = record.to_dict() if isinstance(
            record, ExactCurveDominance) else record
        return provided == exact_g0_curve_dominance(
            objectives, problem=problem, k=k).to_dict()
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return False
