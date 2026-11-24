# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact rational distance bound to a *proved* original DTLZ6 PF point.

Given an EXACT dyadic-rational decision (the numerical float represents
that precise binary rational), preserve its x1 and all K tail coordinates
and snap the inner positions needed for one of the rigorously proved
positive-g support strata. A snapped position is accepted only if the
independent irrational-g global-PF theorem verifies the resulting
exact-real original decision image.

For any two original hyperspherical objectives at the SAME actual g
with only inner coordinates changed, the Euclidean distance is at most

   (pi*g/2) * sum_{inner i} abs(x_i - x'_i).

This follows from the unit norm of every angular partial derivative
and integration along a path changing one inner angle at a time. Use
exact rational Machin pi_upper and exact integer tenth-power bounds on
the IRRATIONAL real DTLZ6 g to obtain an outward rational upper bound.

This is a distance to a proven *feasible Pareto point*. It does NOT
prove exact membership or non-membership of the original source row,
nor that the Pareto point is the nearest point on the entire front.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .dtlz56_rational_interval import (
    _as_exact_fraction, _pi_rational_bounds,
)
from .dtlz56_exact_same_ray import _encode
from .dtlz56_dtlz6_radial_enclosure import (
    certify_dtlz6_radial_interval_pf,
)


@dataclass(frozen=True)
class CertifiedSupportProximity:
    conclusion: str
    dimension: int
    tail_dimension: int
    supporting_objective: int | None
    original_positions: tuple[dict, ...]
    certified_positions: tuple[dict, ...] | None
    tail: tuple[dict, ...]
    certified_l2_distance_upper: dict | None
    supporting_pf_certificate: dict | None
    pi_terms: int
    root_bits: int
    proof_id: str = "DTLZ6-CERTIFIED-SUPPORT-PROXIMITY-V1"
    basis: str = (
        "Exact-real original DTLZ6 hyperspherical angular Lipschitz "
        "bound, rational upper g and pi bounds, and independently "
        "verified global-PF snapped decision"
    )

    def to_dict(self):
        return {
            "conclusion": self.conclusion,
            "dimension": self.dimension,
            "tail_dimension": self.tail_dimension,
            "supporting_objective": self.supporting_objective,
            "original_positions": list(self.original_positions),
            "certified_positions": (
                list(self.certified_positions)
                if self.certified_positions is not None else None
            ),
            "tail": list(self.tail),
            "certified_l2_distance_upper": self.certified_l2_distance_upper,
            "supporting_pf_certificate": self.supporting_pf_certificate,
            "pi_terms": self.pi_terms,
            "root_bits": self.root_bits,
            "proof_id": self.proof_id,
            "basis": self.basis,
            "scientific_certification": "unverified",
        }


def certify_dtlz6_support_proximity(
    positions, tail, *, k, root_bits=40, pi_terms=8,
):
    """Return the least distance bound among finitely checked PF strata.

    A numerical source/objective alignment check is a separate
    prerequisite before applying this conditional decision-image result
    to an archived pair.
    """
    if type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ6 integer K>=1")
    if type(root_bits) is not int or not 4 <= root_bits <= 80:
        raise ValueError("root_bits must be integer in [4,80]")
    if type(pi_terms) is not int or not 2 <= pi_terms <= 32:
        raise ValueError("pi_terms must be integer in [2,32]")
    try:
        u = tuple(_as_exact_fraction(x, name="position") for x in positions)
        t = tuple(_as_exact_fraction(x, name="tail") for x in tail)
    except TypeError as ex:
        raise ValueError("Expected original DTLZ6 decision vectors") from ex
    if len(u) < 3 or any(not 0 <= x <= 1 for x in u):
        raise ValueError("Expected M>=4 and position variables in [0,1]")
    if len(t) != k or any(not 0 <= x <= 1 for x in t):
        raise ValueError("Expected exactly K tail variables in [0,1]")

    m = len(u) + 1
    _, pi_hi = _pi_rational_bounds(pi_terms)
    chosen = None
    for objective in range(1, m-1):
        snapped = list(u)
        if objective == 1:
            for index in range(1, m-1):
                snapped[index] = Fraction(1)
        else:
            sine_position = m - objective
            for index in range(1, sine_position):
                snapped[index] = Fraction(1)
            snapped[sine_position] = Fraction(0)
        snapped = tuple(snapped)
        evidence = certify_dtlz6_radial_interval_pf(
            snapped, t, k=k, root_bits=root_bits, pi_terms=pi_terms,
        )
        if evidence.conclusion != "rigorously_global_pf_member":
            continue
        g_hi = Fraction(
            int(evidence.g_upper["numerator_hex"], 16),
            int(evidence.g_upper["denominator_hex"], 16),
        )
        angular_change = sum(
            (abs(u[i]-snapped[i]) for i in range(1, m-1)),
            Fraction(0),
        )
        distance_hi = pi_hi * g_hi * angular_change / 2
        if chosen is None or distance_hi < chosen[0]:
            chosen = (distance_hi, objective, snapped, evidence.to_dict())

    return CertifiedSupportProximity(
        "rigorously_within_certified_pf_distance"
        if chosen is not None else "unverified",
        m, k, chosen[1] if chosen is not None else None,
        tuple(_encode(x) for x in u),
        (tuple(_encode(x) for x in chosen[2])
         if chosen is not None else None),
        tuple(_encode(x) for x in t),
        _encode(chosen[0]) if chosen is not None else None,
        chosen[3] if chosen is not None else None,
        pi_terms, root_bits,
    )


def replay_dtlz6_support_proximity(
    positions, tail, record, *, k, root_bits=40, pi_terms=8,
):
    """Recalculate the exact bound and embedded PF proof, reject tampering."""
    try:
        supplied = record.to_dict() if isinstance(
            record, CertifiedSupportProximity
        ) else record
        expected = certify_dtlz6_support_proximity(
            positions, tail, k=k, root_bits=root_bits, pi_terms=pi_terms,
        ).to_dict()
        return (
            supplied == expected
            and expected["conclusion"] ==
                "rigorously_within_certified_pf_distance"
        )
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return False
