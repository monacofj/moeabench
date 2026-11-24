# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Symbolic sufficient GLOBAL PF-membership theorem for original DTLZ5/6.

For M>=4, original position coordinates u_1=0, u_2=1, u_3=0,
and any remaining u_i in [0,1], the *feasible* vector at g>=0 is
globally Pareto optimal. In particular, this yields exact positive-g
PF membership for M>=4, previously not established by the other
negative-membership diagnostics.

Proof for g_y>0: y_M=0, forcing any dominating feasible z to theta_1=0.
For every g'<g_y, z_(M-2)=(1+g')cos(theta_2)sin(theta_3)
>= h(g')=(1+g')sin^2(pi/(4(1+g'))).
At the candidate theta_2=b(g_y), theta_3=a(g_y), equality gives
y_(M-2)=h(g_y). Function h is STRICTLY DECREASING for g>=0:
h'(r)=sin(t)*(sin(t)-2*t*cos(t))<0 for t=pi/(4r) in (0,pi/4],
because tan(t)<=4*t/pi<2*t by convexity of tan on [0,pi/4].
Thus z_(M-2)>y_(M-2), impossible for a dominator.
A strict dominator at equal or larger g is excluded by the norm theorem.
The g_y=0 case follows directly from the unit-norm theorem.

This is an exact symbolic theorem, NOT a finite-precision measurement or
an ADR 0059 source/environment/result acceptance certificate.
The input positions and g are exact rational declarations; an external
stored GT row requires independent feasible-image/alignment evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .dtlz56_rational_interval import _as_exact_fraction
from .dtlz56_exact_same_ray import _encode


@dataclass(frozen=True)
class PositiveGGlobalPFCertificate:
    conclusion: str
    dimension: int
    radial_g: dict
    active_positions: tuple[dict, dict, dict]
    proof_id: str = "DTLZ56-M4PLUS-SQUARED-SINE-SUPPORT"
    basis: str = (
        "Exact global norm-order theorem plus strictly decreasing "
        "h(g)=(1+g)sin^2(pi/[4(1+g)]) and theta1=0 objective cap"
    )

    def to_dict(self):
        return {
            "conclusion": self.conclusion,
            "dimension": self.dimension,
            "radial_g": self.radial_g,
            "active_positions": list(self.active_positions),
            "proof_id": self.proof_id,
            "basis": self.basis,
            "scientific_certification": "unverified",
        }


def certify_positive_g_global_pf(positions, *, g, problem, k):
    """Prove global membership on a specified exact original boundary stratum.

    Returns 'rigorously_global_pf_member' only for the sufficient stratum.
    Otherwise returns 'unverified', never 'dominated' or a general decision.

    Original DTLZ5/6 distance ranges are verified from K, and the
    independent position declaration must be aligned to the GT row
    before interpreting the theorem as a statement about that row.
    """
    if problem not in ("DTLZ5", "DTLZ6") or type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ5/6 with integer K>=1")
    try:
        u = tuple(_as_exact_fraction(x, name="position") for x in positions)
    except TypeError as exc:
        raise ValueError("Expected position vector") from exc
    if len(u) < 3 or any(x < 0 or x > 1 for x in u):
        raise ValueError("Expected M>=4 and position variables in [0,1]")
    gy = _as_exact_fraction(g, name="g")
    maximum = Fraction(k, 4) if problem == "DTLZ5" else Fraction(k)
    if gy < 0 or gy > maximum:
        raise ValueError("Distance g is outside original feasible range")
    active = (u[0], u[1], u[2])
    return PositiveGGlobalPFCertificate(
        "rigorously_global_pf_member" if (
            u[0] == 0 and u[1] == 1 and u[2] == 0
        ) else "unverified",
        len(u) + 1, _encode(gy),
        tuple(_encode(v) for v in active),
    )


def replay_positive_g_global_pf(positions, record, *, g, problem, k):
    """Independently check inputs and recompute the exact theorem conditions."""
    try:
        supplied = record.to_dict() if isinstance(
            record, PositiveGGlobalPFCertificate
        ) else record
        independent = certify_positive_g_global_pf(
            positions, g=g, problem=problem, k=k
        ).to_dict()
        return (supplied == independent
                and supplied["conclusion"] == "rigorously_global_pf_member")
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        return False
