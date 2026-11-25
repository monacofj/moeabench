# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tighter EXACT-rational sine upper bound for original DTLZ5/6 PF strata.

The previously proved global support theorem has a sufficient condition
  c**2 < 1 - 4/(p*pi), where c=(1+g)*sin(pi*x1/2), p>=2.
The historical helper bounded sin(theta) by theta. That very
conservative inequality leaves actual DTLZ5 selected M4 rows 7,11
unverified, despite satisfying the mathematical theorem.

For 0<=theta<=pi/2, the alternating Taylor series gives
  sin(theta) <= theta - theta**3/6 + theta**5/120.
For theta_upper=pi_upper*x1/2, theta_upper<2, the upper polynomial
evaluated at theta_upper bounds sin(actual theta), because sin is
monotonic on [0,pi/2] and the Taylor upper bound holds on [0,2].
The entire proof uses exact Fraction calculations with rational Machin
pi bounds; the candidate original g is supplied EXACTLY.

This is a separate opt-in certificate; no regression tests or existing
scientific classifier outputs are modified. The proof concerns exact
original decision-image points and does not certify stored float F or
finite GT adequacy.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .dtlz56_rational_interval import _as_exact_fraction, _pi_rational_bounds
from .dtlz56_generalized_positive_g_pf import _supporting_objective
from .dtlz56_exact_same_ray import _encode


SCHEMA = "DTLZ56-TIGHT-SINE-OPEN-FIRST-ANGLE-V1"


def _taylor_sine_upper(theta: Fraction):
    """Rational upper sin(theta) for 0<=theta<2, no float operations."""
    if not Fraction(0) <= theta < Fraction(2):
        raise ValueError("Taylor upper sine argument must be in [0,2)")
    return theta - theta**3 / 6 + theta**5 / 120


@dataclass(frozen=True)
class TightSineGlobalPFCertificate:
    conclusion: str
    problem: str
    M: int
    K: int
    positions: tuple[dict, ...]
    exact_g: dict
    supporting_objective: int | None
    sine_power: int | None
    pi_terms: int
    sine_theta_upper: dict | None
    last_objective_upper_squared: dict | None
    global_threshold_squared_lower: dict | None
    strict_margin: dict | None
    proof_id: str = SCHEMA

    def to_dict(self):
        return {
            "conclusion": self.conclusion,
            "problem": self.problem,
            "M": self.M, "K": self.K,
            "positions": list(self.positions),
            "exact_g": self.exact_g,
            "supporting_objective": self.supporting_objective,
            "sine_power": self.sine_power,
            "pi_terms": self.pi_terms,
            "sine_theta_upper": self.sine_theta_upper,
            "last_objective_upper_squared":
                self.last_objective_upper_squared,
            "global_threshold_squared_lower":
                self.global_threshold_squared_lower,
            "strict_margin": self.strict_margin,
            "proof_id": self.proof_id,
            "scientific_certification": "unverified",
        }


def certify_tight_sine_global_pf(
    positions, *, g, problem, k, pi_terms=8,
):
    """Prove the earlier global stratum theorem with a sharper sine bound."""
    if problem not in ("DTLZ5", "DTLZ6") or type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ5/6 and integer K>=1")
    if type(pi_terms) is not int or not 2 <= pi_terms <= 32:
        raise ValueError("pi_terms must lie in [2,32]")
    try:
        u = tuple(_as_exact_fraction(v, name="position") for v in positions)
    except TypeError as exc:
        raise ValueError("Expected original positions") from exc
    if len(u) < 3 or any(not Fraction(0) <= x <= 1 for x in u):
        raise ValueError("Expected M>=4 and feasible original positions")
    gy = _as_exact_fraction(g, name="g")
    max_g = Fraction(k, 4) if problem == "DTLZ5" else Fraction(k)
    if not 0 < gy <= max_g:
        raise ValueError("Expected original strictly positive feasible g")

    support, power = _supporting_objective((Fraction(0),) + u[1:])
    theta_upper = c_upper_sq = threshold = margin = None
    certified = False
    if support is not None:
        pi_lo, pi_hi = _pi_rational_bounds(pi_terms)
        if not 3 < pi_lo < pi_hi < 4:
            raise ArithmeticError("Invalid exact Machin pi enclosure")
        theta_upper = pi_hi * u[0] / 2
        sin_upper = _taylor_sine_upper(theta_upper)
        c_upper_sq = ((1 + gy) * sin_upper)**2
        threshold = 1 - Fraction(4, power * pi_lo)
        margin = threshold - c_upper_sq
        certified = threshold > 0 and margin > 0

    return TightSineGlobalPFCertificate(
        "rigorously_global_pf_member" if certified else "unverified",
        problem, len(u)+1, k, tuple(_encode(v) for v in u), _encode(gy),
        support, power, pi_terms,
        _encode(theta_upper) if theta_upper is not None else None,
        _encode(c_upper_sq) if c_upper_sq is not None else None,
        _encode(threshold) if threshold is not None else None,
        _encode(margin) if margin is not None else None,
    )


def replay_tight_sine_global_pf(
    positions, record, *, g, problem, k, pi_terms=8,
):
    """Recompute every exact rational field; no trust in stored conclusions."""
    try:
        supplied = (record.to_dict() if isinstance(
            record, TightSineGlobalPFCertificate) else record)
        expected = certify_tight_sine_global_pf(
            positions, g=g, problem=problem, k=k, pi_terms=pi_terms,
        ).to_dict()
        return (supplied == expected and
                expected["conclusion"] == "rigorously_global_pf_member")
    except (TypeError, ValueError, ArithmeticError, KeyError, ZeroDivisionError):
        return False


def audit_tight_sine_original_rows(
    decisions, objectives, *, problem, m, k, tolerance=1e-9,
):
    """Opt-in companion ledger; excludes any non-aligned or nonrational-g row.

    Only the independently proven exact rational DTLZ5 g is evaluated here.
    DTLZ6 irrational g requires its separately proven enclosure checker.
    """
    import numpy as np

    from .dtlz56_paired_gt_evidence import _original_objectives
    from .dtlz56_paired_symbolic_audit import _exact_radial_g

    if problem != "DTLZ5":
        raise ValueError("Companion ledger is restricted to original DTLZ5")
    x = np.asarray(decisions, dtype=float)
    f = np.asarray(objectives, dtype=float)
    expected, _ = _original_objectives(x, problem=problem, m=m, k=k)
    if f.shape != expected.shape or not np.all(np.isfinite(f)):
        raise ValueError("Require finite, row-aligned objective matrix")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be strictly positive and finite")
    deltas = np.max(np.abs(expected-f), axis=1)
    entries = []
    for i, row in enumerate(x):
        evidence = None
        if deltas[i] <= tolerance:
            gy = _exact_radial_g(row, problem=problem, m=m)
            if gy > 0:
                evidence = certify_tight_sine_global_pf(
                    row[:m-1], g=gy, problem=problem, k=k,
                )
        entries.append({
            "row": i,
            "status": (
                "symbolic_decision_global_pf" if evidence is not None
                and evidence.conclusion == "rigorously_global_pf_member"
                else "unverified"
            ),
            "certificate": (
                evidence.to_dict() if evidence is not None
                and evidence.conclusion == "rigorously_global_pf_member"
                else None
            ),
        })
    certified_rows = [
        e["row"] for e in entries if e["status"] ==
        "symbolic_decision_global_pf"
    ]
    return {
        "schema": "moeabench_dtlz5_tight_sine_row_ledger_v1",
        "problem": problem, "M": m, "K": k,
        "proved_global_decision_image_indices": certified_rows,
        "proved_global_decision_image_count": len(certified_rows),
        "rows": entries,
        "whole_gt_membership": "unverified",
        "finite_coverage": "unverified",
        "scientific_certification": "unverified",
    }
