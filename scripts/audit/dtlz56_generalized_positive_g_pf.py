# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generalized *global* positive-g Pareto membership strata: DTLZ5/6 M>=4.

For each objective j=1,...,M-2, a specific nonempty angular boundary
stratum saturates a lower bound h_p(g)=(1+g)sin(pi/[4(1+g)])**p, p>=2.
The last objective is zero (x1=0). Every strict dominator must have
g'<g from the nonnegative Euclidean norm identity and must also have
x1'=0 from the last objective's cap. Its j-th objective is at least
h_p(g'), which is strictly GREATER than h_p(g). Contradiction.

For j=1: x_2,...,x_(M-1)=1; p=M-2.
For 2<=j<=M-2: x_2,...,x_(M-j)=1 and x_(M-j+1)=0;
                 p=M-j, later positions unrestricted.

This extends the already established j=M-2, x2=1,x3=0 theorem
without modifying its existing implementation or regression tests.

All inputs are *exact rational declarations* and not a certificate that
a stored finite-precision objective row aligns with these decisions.
No production evaluator or GT sampler is invoked.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .dtlz56_rational_interval import _as_exact_fraction
from .dtlz56_exact_same_ray import _encode


@dataclass(frozen=True)
class GeneralizedPositiveGCertificate:
    conclusion: str
    dimension: int
    problem: str
    tail_dimension: int
    radial_g: dict
    supporting_objective: int | None
    sine_power: int | None
    position_declaration: tuple[dict, ...]
    proof_id: str = "DTLZ56-M4PLUS-GLOBAL-PF-SINE-POWER-STRATA-V1"
    basis: str = (
        "Exact nonnegative norm-order theorem; last-objective zero forces "
        "theta1=0; the selected objective saturates globally minimal "
        "(1+g)sin(pi/[4(1+g)])**p with p>=2, strictly decreasing in g"
    )

    def to_dict(self):
        return {
            "conclusion": self.conclusion,
            "dimension": self.dimension,
            "problem": self.problem,
            "tail_dimension": self.tail_dimension,
            "radial_g": self.radial_g,
            "supporting_objective": self.supporting_objective,
            "sine_power": self.sine_power,
            "position_declaration": list(self.position_declaration),
            "proof_id": self.proof_id,
            "basis": self.basis,
            "scientific_certification": "unverified",
        }


def _supporting_objective(positions):
    """Return lowest-index valid supporting objective and its sine power."""
    m = len(positions) + 1
    if positions[0] != 0:
        return None, None
    # j=1: all cos(theta_i) factors, 2 <= i <= M-1, are minimized
    # by their upper original angular boundary (x_i=1).
    if all(v == 1 for v in positions[1:]):
        return 1, m-2
    # j>=2 has M-j-1 constrained cosine factors followed by one sine.
    for j in range(2, m-1):
        sine_index = m-j  # zero-based Python position for x_(M-j+1)
        if positions[sine_index] != 0:
            continue
        if all(positions[index] == 1 for index in range(1, sine_index)):
            return j, m-j
    return None, None


def certify_generalized_positive_g_pf(positions, *, g, problem, k):
    """Exact theorem conditions only; abstain off the listed strata."""
    if problem not in ("DTLZ5", "DTLZ6") or type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ5/6 with integer K>=1")
    try:
        u = tuple(_as_exact_fraction(x, name="position") for x in positions)
    except TypeError as exc:
        raise ValueError("Expected position vector") from exc
    if len(u) < 3 or any(x < 0 or x > 1 for x in u):
        raise ValueError("Expected M>=4 and positions in [0,1]")
    gy = _as_exact_fraction(g, name="g")
    gmax = Fraction(k, 4) if problem == "DTLZ5" else Fraction(k)
    if not 0 <= gy <= gmax:
        raise ValueError("Distance g outside original feasible range")

    objective, power = _supporting_objective(u)
    return GeneralizedPositiveGCertificate(
        "rigorously_global_pf_member" if objective is not None else "unverified",
        len(u)+1, problem, k, _encode(gy), objective, power,
        tuple(_encode(x) for x in u),
    )


def replay_generalized_positive_g_pf(positions, record, *, g, problem, k):
    """Recompute every exact stratum condition; reject forged conclusions."""
    try:
        supplied = record.to_dict() if isinstance(
            record, GeneralizedPositiveGCertificate
        ) else record
        expected = certify_generalized_positive_g_pf(
            positions, g=g, problem=problem, k=k
        ).to_dict()
        return (supplied == expected
                and expected["conclusion"] == "rigorously_global_pf_member")
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return False
