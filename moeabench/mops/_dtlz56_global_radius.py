# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent scalar-radius global dominance criterion for DTLZ5 and DTLZ6.

Mathematical equivalence for any feasible y in the original problems:

    y is Pareto-optimal  iff
    no feasible z <= y has g(z) < g(y).

All feasible objectives obey ||f||_2 = 1 + g. If z <= y strictly in
one coordinate, then ||z||_2 < ||y||_2, hence g(z) < g(y).
Conversely, any feasible z <= y with smaller g strictly dominates y.

This implementation numerically checks a *closed, margin-separated*
subrange 0 <= g <= g(y)-g_gap using optimistic continuous-g interval
relaxations and independently reconstructed feasible witnesses.
Successful numerical exclusion is NOT an exact global Pareto proof:
arbitrarily small improvements and non-directed floating-point error
remain outside its claim. No production PF/PS, scalarizer or evaluator
is invoked.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin, sqrt
import numpy as np

from ._dtlz56_dominance_bounds import _relaxed_patch_can_fit
from ._dtlz56_independent_audit import inspect_dtlz56_geometry


@dataclass(frozen=True)
class GlobalRadiusAudit:
    status: str
    g_candidate: float
    g_gap: float
    g_checked_upper: float
    intervals_tested: int
    intervals_unresolved: int
    max_intervals: int
    witness_g: float | None = None
    witness_objectives: tuple[float, ...] | None = None

    def to_dict(self):
        return {
            "status": self.status,
            "g_candidate": self.g_candidate,
            "g_gap": self.g_gap,
            "g_checked_upper": self.g_checked_upper,
            "intervals_tested": self.intervals_tested,
            "intervals_unresolved": self.intervals_unresolved,
            "max_intervals": self.max_intervals,
            "witness_g": self.witness_g,
            "witness_objectives": (
                list(self.witness_objectives)
                if self.witness_objectives is not None else None
            ),
            "basis": (
                "Independent original-DTLZ5/6 scalar-radius dominance "
                "equivalence and optimistic continuous-g patch tests; "
                "finite g gap and ordinary FP are not an exact proof"
            ),
        }


def _fixed_g_witness(caps, g, *, tolerance):
    """Construct a feasible original angular-patch vector <= caps, if found.

    The independent greedy back-substitution allocates the maximum
    admissible contribution to each objective, right-to-left.
    A returned vector is checked against angular constraints and the
    original unit-sphere equation before it is used as evidence.
    """
    m = len(caps)
    r = 1.0+g
    if np.any(caps < -tolerance):
        return None
    if caps[-1] >= r:
        result = np.zeros(m)
        result[-1] = r
        return result
    if caps[-1] < 0:
        return None

    out = np.zeros(m)
    out[-1] = min(r, float(caps[-1]))
    rest = sqrt(max(0.0, r*r-out[-1]*out[-1]))
    lo = pi/(4*(1+g))
    hi = pi*(1+2*g)/(4*(1+g))
    for j in range(m-2, 0, -1):
        if rest*sin(lo) > caps[j]+tolerance:
            return None
        out[j] = min(rest*sin(hi), max(0.0, float(caps[j])))
        rest = sqrt(max(0.0, rest*rest-out[j]*out[j]))
    if rest > caps[0]+tolerance:
        return None
    out[0] = rest
    if np.any(out > caps+tolerance):
        return None
    if abs(np.linalg.norm(out)-r) > 8*tolerance:
        return None
    return out


def audit_global_radius_dominance(
    objective, *, problem, k, g_gap=1e-4, max_intervals=8192,
    numerical_slack=1e-11,
):
    """Numerically exclude g <= g(y)-g_gap or find a strict dominator.

    "witness" provides an independently reconstructed feasible vector
    below y, confirmed within stated ordinary-FP tolerance. "excluded"
    excludes only dominators whose radial g improvement is >=g_gap.
    "unverified" means insufficient interval/precision evidence.
    """
    y = np.asarray(objective, dtype=float)
    if y.ndim != 1 or len(y) < 2 or not np.all(np.isfinite(y)):
        raise ValueError("objective must be a finite 1-D objective vector")
    if np.any(y < 0):
        raise ValueError("objective coordinates must be nonnegative")
    if problem not in ("DTLZ5", "DTLZ6") or not isinstance(k, int) or k < 1:
        raise ValueError("Expected original DTLZ5/6 with K >= 1")
    if not np.isfinite(g_gap) or g_gap <= 0:
        raise ValueError("g_gap must be positive and finite")
    if not isinstance(max_intervals, int) or max_intervals < 1:
        raise ValueError("max_intervals must be a positive integer")
    if not np.isfinite(numerical_slack) or numerical_slack <= 0:
        raise ValueError("numerical_slack must be positive and finite")

    geometry = inspect_dtlz56_geometry(
        y[None, :], m=len(y), k=k, problem=problem,
        tolerance=numerical_slack)
    if geometry["outside_feasible_image"]:
        raise ValueError("candidate is not in the original feasible image")

    gy = float(np.linalg.norm(y)-1)
    upper = max(0.0, gy-float(g_gap))
    if upper == 0.0:
        return GlobalRadiusAudit(
            "excluded", gy, float(g_gap), 0.0, 0, 0,
            max_intervals)

    queue = [(0.0, upper)]
    intervals = 0
    unresolved = 0
    while queue:
        left, right = queue.pop()
        if intervals >= max_intervals:
            unresolved += 1+len(queue)
            break
        intervals += 1

        # At any g in [left, right], angular domains are nested inside
        # the g=right domain and radius >= 1+left. This interval relaxation
        # is optimistic for the existence of a dominating vector.
        radius = max(0.0, 1+left-2*numerical_slack)
        if not _relaxed_patch_can_fit(
            y, radius, right, slack=numerical_slack
        ):
            continue

        # Independently reconstruct an actual feasible vector at fixed
        # g; do not accept optimistic interval feasibility as a witness.
        for g in (left, (left+right)/2, right):
            candidate = _fixed_g_witness(y, g, tolerance=numerical_slack)
            if candidate is None:
                continue
            residual = inspect_dtlz56_geometry(
                candidate[None, :], m=len(y), k=k, problem=problem,
                tolerance=16*numerical_slack)
            if residual["outside_feasible_image"] != 0:
                continue
            if (np.all(candidate <= y+numerical_slack)
                    and np.linalg.norm(candidate) <
                    np.linalg.norm(y)-g_gap/2):
                return GlobalRadiusAudit(
                    "witness", gy, float(g_gap), upper,
                    intervals, 0, max_intervals, float(g),
                    tuple(map(float, candidate)))

        midpoint = (left+right)/2
        if midpoint == left or midpoint == right:
            unresolved += 1
            continue
        queue.extend(((left, midpoint), (midpoint, right)))

    return GlobalRadiusAudit(
        "unverified" if unresolved else "excluded",
        gy, float(g_gap), upper, intervals, unresolved, max_intervals)
