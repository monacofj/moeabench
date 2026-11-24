# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent objective-to-decision reconstruction for original DTLZ5/6.

An objective-only GT does not store a position vector. This diagnostic
reconstructs a canonical feasible decision from the ORIGINAL published
spherical equations, verifies it independently against the claimed
objective row, and then requests the pre-existing one-sided local
dominance audit. It can find a dominated M>=4 candidate but CANNOT
certify global Pareto membership when no local witness is found.

Ordinary floating-point evidence is provisional under ADR 0059.
No production evaluator, Pareto sampler or scalarizer is called.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi, sqrt

import numpy as np

from ._dtlz56_independent_audit import inspect_dtlz56_geometry
from ._dtlz56_local_descent import (
    _objective_and_derivative,
    _reconstruct_decision,
    audit_local_dtlz56_descent,
)


@dataclass(frozen=True)
class ObjectiveReconstructionAudit:
    status: str
    reconstruction_error: float | None
    reconstructed_decision: tuple[float, ...] | None
    local_witness: dict | None
    basis: str = (
        "Independent inverse spherical equations and original g tail; "
        "aligned reconstructed objective checked before local-descent "
        "search; absence of witness is not global nondominance proof"
    )

    def to_dict(self):
        return {
            "status": self.status,
            "reconstruction_error": self.reconstruction_error,
            "reconstructed_decision": (
                None if self.reconstructed_decision is None
                else list(self.reconstructed_decision)
            ),
            "local_witness": self.local_witness,
            "basis": self.basis,
        }


def audit_objective_local_descent(
    objective, *, problem, k, tolerance=1e-9, max_halvings=30
):
    """Find a provisional strict dominator without supplied PS witnesses.

    'witness' means the independent local audit reconstructed a smaller-g
    feasible candidate that numerically improves every objective.
    'unverified' covers ambiguity, failures and non-detection, never PASS.
    Invalid shape/problem/negative coordinate inputs raise ValueError.
    """
    y = np.asarray(objective, dtype=float)
    if y.ndim != 1 or y.size < 2 or not np.isfinite(y).all():
        raise ValueError("Expected finite 1-D objective vector, M>=2")
    if np.any(y < 0):
        raise ValueError("Expected nonnegative objectives")
    if problem not in ("DTLZ5", "DTLZ6") or type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ5/6 and integer K>=1")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Expected positive finite tolerance")
    if type(max_halvings) is not int or max_halvings < 1:
        raise ValueError("Expected positive integer max_halvings")

    m = y.size
    result = inspect_dtlz56_geometry(
        y[None, :], m=m, k=k, problem=problem, tolerance=tolerance
    )
    if result["outside_feasible_image"]:
        raise ValueError("Candidate is outside the original feasible image")

    radius = float(np.linalg.norm(y))
    g = radius-1.0
    if g <= tolerance:
        return ObjectiveReconstructionAudit("unverified", None, None, None)

    # theta_1 = atan2(f_M, norm(f_1, ..., f_(M-1))).
    # For i>=2 the inverse angles are determined except at exact 0/0
    # singularities, where pi/4 is always realizable.
    u = np.empty(m-1, dtype=float)
    u[0] = 2.0*atan2(float(y[-1]), float(np.linalg.norm(y[:-1])))/pi
    for i in range(1, m-1):
        idx = m-i-1
        numerator = float(y[idx])
        subradius = float(np.linalg.norm(y[:idx]))
        theta = (pi/4 if numerator == 0.0 and subradius == 0.0
                 else atan2(numerator, subradius))
        u[i] = (4.0*(1.0+g)*theta/pi-1.0)/(2.0*g)

    # Provisional floating-point inverse: values outside the feasible box
    # by more than tolerance are not silently projected into it.
    if np.any(u < -tolerance) or np.any(u > 1.0+tolerance):
        return ObjectiveReconstructionAudit("unverified", None, None, None)
    u = np.clip(u, 0.0, 1.0)

    reconstructed, _, _ = _objective_and_derivative(u, g)
    error = float(np.max(np.abs(reconstructed-y)))
    if error > tolerance or not np.isfinite(error):
        return ObjectiveReconstructionAudit("unverified", error, None, None)

    decision = _reconstruct_decision(u, g, problem=problem, k=k)
    if np.any(decision < 0) or np.any(decision > 1):
        return ObjectiveReconstructionAudit("unverified", error, None, None)
    # Check that the actual stored decision tail recreates the target g;
    # DTLZ6's inverse exponent can underflow for tiny positive g.
    tail = decision[m-1:]
    actual_g = (float(np.sum((tail-.5)**2)) if problem == "DTLZ5"
                else float(np.sum(tail**0.1)))
    if abs(actual_g-g) > tolerance:
        return ObjectiveReconstructionAudit("unverified", error, None, None)

    local = audit_local_dtlz56_descent(
        decision, problem=problem, m=m, numerical_tolerance=tolerance,
        max_halvings=max_halvings,
    )
    # Independently check the found witness against original objective
    # caps, rather than transferring the local audit result blindly.
    status = "unverified"
    if local.status == "witness" and local.witness_objectives is not None:
        z = np.asarray(local.witness_objectives, dtype=float)
        if (np.all(z <= y-tolerance)
                and np.linalg.norm(z) < radius-tolerance):
            status = "witness"
    return ObjectiveReconstructionAudit(
        status, error, tuple(map(float, decision)), local.to_dict()
    )
