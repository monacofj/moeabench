# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent continuous-g bounded-margin dominance exclusion for DTLZ5/6.

For an objective vector y, exclude all feasible z <= y with
z[j] <= y[j]-margin for at least one j, by independent scalar-g
interval branch-and-bound. Excluding a margin is not a proof of
exact Pareto optimality.

For each g interval [L,U], use the optimistic smaller radius 1+L
and wider angular interval at U. A hypothetical feasible dominating
z within [L,U], radially contracted to 1+L, would remain inside this
relaxed patch and satisfy the same coordinate caps. Therefore
infeasibility of the relaxation excludes the entire continuous
interval in exact arithmetic.

Ordinary floating-point computations with outward safety slack are
NOT a formally directed-rounded interval proof. The result requires
independent numerical/environment verification under ADR 0059.
No production scalarizer, PS, PF or evaluator is used here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DeltaDominanceAudit:
    status: str
    margin: float
    checked_objectives: int
    excluded_objectives: int
    intervals_tested: int
    intervals_unresolved: int
    max_intervals: int
    basis: str = (
        "Independent optimistic g-interval spherical-patch feasibility, "
        "with numerical safety slack; not exact Pareto proof"
    )

    def to_dict(self):
        return {
            "status": self.status, "margin": self.margin,
            "checked_objectives": self.checked_objectives,
            "excluded_objectives": self.excluded_objectives,
            "intervals_tested": self.intervals_tested,
            "intervals_unresolved": self.intervals_unresolved,
            "max_intervals": self.max_intervals,
            "basis": self.basis,
        }


def _relaxed_patch_can_fit(caps, radius, angular_g, *, slack):
    """Optimistic coordinate-box feasibility of spherical angular patch."""
    if np.any(caps < -slack):
        return False
    if caps[-1] >= radius - slack:
        return True
    # theta_1 is freely available in [0, pi/2].
    last = min(radius, max(0.0, float(caps[-1]) + slack))
    remaining = math.sqrt(max(0.0, radius*radius-last*last))
    lower = math.pi/(4.0*(1.0+angular_g))
    upper = math.pi*(1.0+2.0*angular_g)/(4.0*(1.0+angular_g))
    lower = max(0.0, lower-slack)
    upper = min(math.pi/2, upper+slack)
    for j in range(len(caps)-2, 0, -1):
        if remaining*math.sin(lower) > caps[j] + 4*slack:
            return False
        contribution = min(remaining*math.sin(upper),
                           max(0.0, float(caps[j])+slack))
        remaining = math.sqrt(max(0.0, remaining**2-contribution**2))
    return remaining <= caps[0]+4*slack


def audit_delta_dominance(
    objective, *, problem, k, margin, max_intervals=4096,
    numerical_slack=1e-11,
):
    """Exclude objective-wise improvements >=margin over all feasible g.

    Return excluded only if the optimistic relaxation excludes every
    coordinate improvement over the full continuous g domain.
    Unknown, unresolved and budget-exhausted cases return unverified.
    """
    y = np.asarray(objective, dtype=float)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("Expected one finite objective vector")
    if np.any(y < 0):
        raise ValueError("Expected a nonnegative objective vector")
    if problem not in ("DTLZ5", "DTLZ6") or k < 1:
        raise ValueError("Expected original DTLZ5/6 and K >= 1")
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    if not isinstance(max_intervals, int) or max_intervals < 1:
        raise ValueError("max_intervals must be an integer >= 1")
    if not np.isfinite(numerical_slack) or numerical_slack <= 0:
        raise ValueError("numerical_slack must be finite and positive")
    gmax = k/4.0 if problem == "DTLZ5" else float(k)
    tested = excluded = unresolved = 0
    for j in range(len(y)):
        caps = y.copy()
        caps[j] -= margin
        if caps[j] < -numerical_slack:
            excluded += 1
            continue
        queue = [(0.0, gmax)]
        pending = 0
        while queue:
            left, right = queue.pop()
            if tested >= max_intervals:
                pending += 1+len(queue)
                break
            tested += 1
            optimistic_radius = max(0.0, 1.0+left-2*numerical_slack)
            if not _relaxed_patch_can_fit(
                    caps, optimistic_radius, right, slack=numerical_slack):
                continue
            middle = left+(right-left)/2
            if middle == left or middle == right:
                pending += 1
                continue
            queue.extend(((left, middle), (middle, right)))
        if pending:
            unresolved += pending
        else:
            excluded += 1
    return DeltaDominanceAudit(
        status="excluded" if excluded == len(y) else "unverified",
        margin=float(margin), checked_objectives=len(y),
        excluded_objectives=excluded, intervals_tested=tested,
        intervals_unresolved=unresolved, max_intervals=max_intervals,
    )
