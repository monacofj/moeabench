# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Global positive-g Pareto certificate with a nonzero first angle.

Original DTLZ5/6, M>=4. The previous global-PF supporting strata require
x1=0. Here their same supporting angular boundary patterns are extended
to a rigorously bounded OPEN range of x1>0, using exact-rational
upper/lower enclosures of pi and no floating-point trigonometry.

Let r=1+g, c=r*sin(pi*x1/2), and choose a supporting objective j with
p>=2 sine/cosine factors minimized at the original inner bounds.
The candidate has y_M=c, y_j=sqrt(r*r-c*c)*sin^p(pi/(4r)).
Any strict dominator z has 1<=r'<r and z_M<=c; hence
z_j>=H_p(r';c)=sqrt(r'*r'-c*c)*sin^p(pi/(4r')).
For c<1 and every r'>=1:
  r'^2/(r'^2-c^2) <= 1/(1-c^2),
  p*t*cot(t)>=p*pi/4, where t=pi/(4r').
Thus if c^2 < 1-4/(p*pi), derivative H_p'(r';c)<0
throughout r'>=1. Hence z_j>y_j, contradiction.

This checker uses B=r*pi_upper*x1/2 >= c (sin theta <= theta)
and the provably more restrictive EXACT RATIONAL condition
  B^2 < 1-4/(p*pi_lower).
The Machin alternating series provides rational pi bounds,
so every accepted conclusion is mathematically sound. Rejection
merely means this sufficient theorem could not be applied.

Its conditional theorem presumes that the supplied original position
declaration and g correspond to a feasible candidate. It does not
certify a stored objective-only GT row, scientific environment,
reference coverage or indicator suitability (ADR 0059).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .dtlz56_exact_same_ray import _encode
from .dtlz56_generalized_positive_g_pf import _supporting_objective
from .dtlz56_rational_interval import _as_exact_fraction, _pi_rational_bounds


@dataclass(frozen=True)
class OpenFirstAnglePFCertificate:
    conclusion: str
    problem: str
    dimension: int
    tail_dimension: int
    original_g: dict
    position_declaration: tuple[dict, ...]
    supporting_objective: int | None
    sine_power: int | None
    pi_terms: int
    upper_last_objective_squared: dict | None
    safe_last_objective_threshold_squared: dict | None
    strict_rational_margin: dict | None
    proof_id: str = "DTLZ56-POSITIVE-G-OPEN-FIRST-ANGLE-H-POWER-V1"
    basis: str = (
        "Exact Machin rational pi bounds, sin(theta)<=theta, "
        "strict monotonicity of sqrt(r²-c²)*sin^p(pi/[4r]) on r>=1"
    )

    def to_dict(self):
        return {
            "conclusion": self.conclusion,
            "problem": self.problem,
            "dimension": self.dimension,
            "tail_dimension": self.tail_dimension,
            "original_g": self.original_g,
            "position_declaration": list(self.position_declaration),
            "supporting_objective": self.supporting_objective,
            "sine_power": self.sine_power,
            "pi_terms": self.pi_terms,
            "upper_last_objective_squared": self.upper_last_objective_squared,
            "safe_last_objective_threshold_squared":
                self.safe_last_objective_threshold_squared,
            "strict_rational_margin": self.strict_rational_margin,
            "proof_id": self.proof_id,
            "basis": self.basis,
            "scientific_certification": "unverified",
        }


def certify_open_first_angle_global_pf(
    positions, *, g, problem, k, pi_terms=8,
):
    """Sound positive global membership on supporting strata near x1=0.

    Returns only 'rigorously_global_pf_member' (strict rational proof)
    or 'unverified'. No finite-precision numerical tolerance is used.
    """
    if problem not in ("DTLZ5", "DTLZ6") or type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ5/6 with K>=1")
    if type(pi_terms) is not int or not 2 <= pi_terms <= 32:
        raise ValueError("pi_terms must be an integer in [2,32]")
    try:
        u = tuple(_as_exact_fraction(v, name="position") for v in positions)
    except TypeError as exc:
        raise ValueError("Expected position vector") from exc
    if len(u) < 3 or any(v < 0 or v > 1 for v in u):
        raise ValueError("Expected M>=4 and position variables in [0,1]")
    gy = _as_exact_fraction(g, name="g")
    gmax = Fraction(k, 4) if problem == "DTLZ5" else Fraction(k)
    if not 0 <= gy <= gmax:
        raise ValueError("Distance g is outside original feasible range")

    # Existing theorem's angular support patterns, but deliberately
    # remove x1=0 restriction: the present proof handles x1>0.
    support_positions = (Fraction(0),) + u[1:]
    j, p = _supporting_objective(support_positions)
    bound_sq = threshold = margin = None
    approved = False
    if j is not None:
        pi_lo, pi_hi = _pi_rational_bounds(pi_terms)
        if not (3 < pi_lo < pi_hi < 4):
            raise ArithmeticError("Rational Machin enclosure invalid")
        r = 1 + gy
        # sin(pi*x1/2) <= pi*x1/2, so c^2 cannot exceed bound_sq.
        bound_sq = (r*pi_hi*u[0]/2)**2
        threshold = 1 - Fraction(4, p*pi_lo)
        margin = threshold - bound_sq
        approved = (threshold > 0 and margin > 0)

    return OpenFirstAnglePFCertificate(
        "rigorously_global_pf_member" if approved else "unverified",
        problem, len(u)+1, k, _encode(gy),
        tuple(_encode(v) for v in u), j, p, pi_terms,
        _encode(bound_sq) if bound_sq is not None else None,
        _encode(threshold) if threshold is not None else None,
        _encode(margin) if margin is not None else None,
    )


def replay_open_first_angle_global_pf(
    positions, record, *, g, problem, k, pi_terms=8,
):
    """Recompute strict exact-rational proof and refuse amended certificates."""
    try:
        supplied = (
            record.to_dict() if isinstance(record, OpenFirstAnglePFCertificate)
            else record
        )
        expected = certify_open_first_angle_global_pf(
            positions, g=g, problem=problem, k=k, pi_terms=pi_terms
        ).to_dict()
        return (
            supplied == expected
            and expected["conclusion"] == "rigorously_global_pf_member"
        )
    except (ValueError, KeyError, TypeError, ArithmeticError,
            ZeroDivisionError):
        return False
