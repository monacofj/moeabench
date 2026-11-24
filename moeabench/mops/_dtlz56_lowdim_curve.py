# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent zero-gap *negative* Pareto-membership witness for M=2/3 DTLZ5/6.

Mathematical theorem (original problems, exact real arithmetic):
Every feasible objective vector with g>0 and M in {2, 3} is dominated
by a feasible vector on the complete g=0 curve, without choosing a
positive margin in g. This does not describe the M>=4 front.

The numerical witness is deliberately provisional (ADR 0059): ordinary
float evaluations of square roots/trigonometry cannot certify delicate
boundary inequalities without independent error enclosures.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import numpy as np

from ._dtlz56_independent_audit import inspect_dtlz56_geometry


@dataclass(frozen=True)
class LowDimensionalCurveAudit:
    status: str
    candidate_g: float
    witness: tuple[float, ...] | None
    minimum_slack: float
    basis: str = (
        "Original DTLZ5/6 M=2/3: exact concavity proof that every "
        "feasible positive-g vector is dominated by a same-theta1 "
        "g=0 curve point; numeric witness remains provisional"
    )

    def to_dict(self):
        return {
            "status": self.status,
            "candidate_g": self.candidate_g,
            "witness": None if self.witness is None else list(self.witness),
            "minimum_slack": self.minimum_slack,
            "basis": self.basis,
        }


def audit_low_dimensional_g0_dominator(
    objective, *, problem, k, tolerance=1e-9
):
    """Construct an independent g=0 dominator for original M=2/3.

    The caller supplies an independently feasible objective vector.
    A returned 'witness' means a numerically separated dominance
    candidate, NOT a certified cross-platform floating-point proof.
    At g=0 or insufficient separation the status is 'unverified'.
    """
    y = np.asarray(objective, dtype=float)
    if y.ndim != 1 or y.size not in (2, 3) or not np.isfinite(y).all():
        raise ValueError("Expected finite M=2 or M=3 objective vector")
    if np.any(y < 0):
        raise ValueError("Expected nonnegative objectives")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if problem not in ("DTLZ5", "DTLZ6") or not isinstance(k, int) or k < 1:
        raise ValueError("Expected original DTLZ5/6 and K>=1")

    geometry = inspect_dtlz56_geometry(
        y[None, :], m=y.size, k=k, problem=problem, tolerance=tolerance
    )
    if geometry["outside_feasible_image"]:
        raise ValueError("Candidate is outside original feasible image")

    radius = float(np.linalg.norm(y))
    gy = radius - 1.0
    # Matching theta1 means sin(theta1) = y_M / (1+g).
    last = min(1.0, float(y[-1]) / radius)
    common = sqrt(max(0.0, 1.0-last*last))
    if y.size == 2:
        z = np.asarray([common, last])
    else:
        z = np.asarray([common/sqrt(2.0), common/sqrt(2.0), last])
    slack = y-z
    # Boundary equality is expected for an objective on a zero axis.
    # Require at least one safely strict improvement and no coordinate
    # increase beyond tolerance, without asserting machine certification.
    safely_dominated = (
        gy > tolerance
        and np.all(slack >= -tolerance)
        and np.any(slack > tolerance)
        and float(np.linalg.norm(z)) < radius - tolerance
    )
    return LowDimensionalCurveAudit(
        "witness" if safely_dominated else "unverified",
        gy,
        tuple(map(float, z)) if safely_dominated else None,
        float(np.min(slack)),
    )
