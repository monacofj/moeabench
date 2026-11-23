# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent complete g=0 Pareto-curve dominance for original DTLZ5/6.

The exact criterion is documented in docs/gt.md, Theorem 6:
the g=0 curve C(t) weakly dominates nonnegative y iff
y_j >= a_j * sqrt(1-min(y_M,1)^2) for all j<M.
The candidate at t=asin(min(y_M,1)) is constructive.

This is independent of production pf, ps, evaluation and scalarizer.
Floating-point witnesses are provisional numerical evidence until
verified under the ADR 0059 source/environment/result gates.
A negative result only excludes g=0 *curve* dominance, not all
positive-g feasible dominators.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import numpy as np


@dataclass(frozen=True)
class CurveDominanceAudit:
    status: str
    candidate_g: float
    min_objective_slack: float
    witness: tuple[float, ...] | None
    basis: str = (
        "Exact complete g=0 curve dominance theorem; constructive "
        "independent numeric witness with explicit precision margin; "
        "absence of curve dominance is not global Pareto certification"
    )

    def to_dict(self):
        return {
            "status": self.status,
            "candidate_g": self.candidate_g,
            "min_objective_slack": self.min_objective_slack,
            "witness": None if self.witness is None else list(self.witness),
            "basis": self.basis,
        }


def audit_g0_curve_dominance(objective, *, tolerance=1e-9):
    """Check all continuous g=0 curve points, without sampling the curve.

    'witness' requires a positive safety margin on every first M-1
    coordinate, and positive g. This deliberately excludes delicate
    boundary-equality cases rather than claiming an unsafe certificate.
    'no_curve_dominator' is an exact-theorem conclusion conditioned
    on numerical tolerance, not a global Pareto claim.
    'unverified' flags numerical equality or potential singularity.
    """
    y = np.asarray(objective, dtype=float)
    if y.ndim != 1 or y.size < 2 or not np.isfinite(y).all():
        raise ValueError("Expected finite M>=2 objective vector")
    if np.any(y < 0):
        raise ValueError("Expected nonnegative objective coordinates")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    m = len(y)
    coefficients = np.asarray(
        [2.0**(-(m-2)/2)] +
        [2.0**(-(m-j)/2) for j in range(2, m)],
        dtype=float,
    )
    last = min(float(y[-1]), 1.0)
    first = coefficients * sqrt(max(0.0, 1.0-last*last))
    candidate = np.concatenate((first, [last]))
    slack = y[:-1] - first
    minimum = float(np.min(slack))
    gy = float(np.linalg.norm(y)-1.0)

    # The mathematical witness has z_M=y_M when y_M<=1, and all
    # remaining coordinates are strictly smaller with safety margin.
    # Thus the proof is not relying on small float differences at the
    # very coordinate used to select the curve parameter.
    if gy > tolerance and minimum > tolerance:
        return CurveDominanceAudit(
            "witness", gy, minimum, tuple(map(float, candidate)))
    if minimum < -tolerance:
        return CurveDominanceAudit(
            "no_curve_dominator", gy, minimum, None)
    return CurveDominanceAudit("unverified", gy, minimum, None)
