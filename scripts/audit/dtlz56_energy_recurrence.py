# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent squared-radius fixed-g cap feasibility for original DTLZ5/6.

In exact real arithmetic, this O(M) recurrence is equivalent to the
published spherical-patch dominance predicate D(g; y).

Unlike the angle-allocation recursion it uses no repeated square roots.
That is useful for future outward interval enclosures. Here all arithmetic
(including sin) is still ordinary floating point: reported feasible or
infeasible observations MUST NOT be promoted to ADR 0059 certification.
No production evaluator, PS/PF, scalarizer or optimization is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sin

import numpy as np


@dataclass(frozen=True)
class SquaredRadiusFeasibility:
    observation: str
    g: float
    beta: float
    residual_squared: float
    stopping_objective: int | None
    numerical_gap: float
    basis: str = (
        "Exact-real fixed-g squared-energy recurrence; "
        "ordinary floating-point observation is not a certified enclosure"
    )

    def to_dict(self):
        return {
            "observation": self.observation,
            "g": self.g,
            "beta": self.beta,
            "residual_squared": self.residual_squared,
            "stopping_objective": self.stopping_objective,
            "numerical_gap": self.numerical_gap,
            "basis": self.basis,
            "scientific_status": "unverified",
        }


def observe_fixed_g_energy_feasibility(
    objectives, *, g, problem, k
) -> SquaredRadiusFeasibility:
    """Check coordinate caps at exactly one g, not over an interval of g.

    With r=1+g and beta=sin(pi/(4r))**2, the minimum feasible
    contribution of each constrained inner angle uses q*beta.
    If its squared objective cap is insufficient the patch cannot fit.
    Otherwise greedily taking the maximum admissible contribution leaves
    q_new = max(q*beta, q-cap**2).

    Both comparisons and all returned outcomes are *numerical* until a
    directed-rounded (or equally rigorous) implementation is available.
    """
    y = np.asarray(objectives, dtype=float)
    if y.ndim != 1 or y.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("Expected a finite objective vector with M>=2")
    if np.any(y < 0):
        raise ValueError("Objective caps must be nonnegative")
    if problem not in ("DTLZ5", "DTLZ6") or type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ5/6 and integer K>=1")
    if not isfinite(g) or g < 0:
        raise ValueError("Expected nonnegative finite g")
    maximum = k / 4.0 if problem == "DTLZ5" else float(k)
    if g > maximum:
        raise ValueError("g outside original feasible distance range")

    r = 1.0 + float(g)
    beta = sin(pi / (4.0 * r))**2
    if y[-1] >= r:
        # theta_1=pi/2: all preceding objective coordinates vanish.
        return SquaredRadiusFeasibility(
            "numerically_feasible", float(g), beta, 0.0, None,
            float(y[-1] - r),
        )

    q = r*r - float(y[-1])**2
    for j in range(len(y)-2, 0, -1):
        minimum_squared = q*beta
        cap_squared = float(y[j])**2
        if minimum_squared > cap_squared:
            return SquaredRadiusFeasibility(
                "numerically_infeasible", float(g), beta, q, j+1,
                float(cap_squared - minimum_squared),
            )
        q = max(minimum_squared, q-cap_squared)

    gap = float(y[0])**2 - q
    return SquaredRadiusFeasibility(
        "numerically_feasible" if gap >= 0 else "numerically_infeasible",
        float(g), beta, q,
        None if gap >= 0 else 1, gap,
    )


@dataclass(frozen=True)
class SquaredIntervalExclusion:
    observation: str
    g_lower: float
    g_upper: float
    beta_lower: float
    residual_lower_squared: float
    stopping_objective: int | None
    numerical_gap: float
    basis: str = (
        "Exact-real optimistic continuous-g squared-energy lower bound; "
        "ordinary FP cannot provide a rigorous outward enclosure"
    )

    def to_dict(self):
        return {
            "observation": self.observation,
            "g_lower": self.g_lower,
            "g_upper": self.g_upper,
            "beta_lower": self.beta_lower,
            "residual_lower_squared": self.residual_lower_squared,
            "stopping_objective": self.stopping_objective,
            "numerical_gap": self.numerical_gap,
            "basis": self.basis,
            "scientific_status": "unverified",
        }


def observe_interval_energy_exclusion(
    objectives, *, g_lower, g_upper, problem, k
) -> SquaredIntervalExclusion:
    """Optimistically exclude all fixed-g feasibilities in [L,U].

    Exact-real proof: r(g)^2 is increasing and beta(g) decreasing.
    For all g in [L,U], q0(g)>=max(0,(1+L)^2-y_M^2) and
    beta(g)>=sin(pi/(4(1+U)))**2. The functions
    max(q*beta,q-cap**2) and q*beta are nondecreasing in both
    nonnegative q and beta. Propagate lower q bounds; only an
    infeasible cap for *every* g can produce an exclusion.
    Ordinary float output is a provisional observation, not a
    directed-rounded continuous-interval scientific certificate.
    """
    y = np.asarray(objectives, dtype=float)
    if y.ndim != 1 or y.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("Expected finite objective vector M>=2")
    if np.any(y < 0):
        raise ValueError("Objective caps must be nonnegative")
    if problem not in ("DTLZ5", "DTLZ6") or type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ5/6, integer K>=1")
    maximum = k / 4.0 if problem == "DTLZ5" else float(k)
    if (not isfinite(g_lower) or not isfinite(g_upper) or
            g_lower < 0 or g_lower > g_upper or g_upper > maximum):
        raise ValueError("Expected 0 <= g_lower <= g_upper <= original g_max")
    lower, upper = float(g_lower), float(g_upper)
    beta_lower = sin(pi/(4*(1+upper)))**2
    q_lower = max(0.0, (1+lower)**2-float(y[-1])**2)
    for j in range(len(y)-2, 0, -1):
        margin = float(y[j])**2-q_lower*beta_lower
        if margin < 0:
            return SquaredIntervalExclusion(
                "numerically_excluded", lower, upper, beta_lower,
                q_lower, j+1, margin,
            )
        q_lower = max(q_lower*beta_lower, q_lower-float(y[j])**2)
    final_margin = float(y[0])**2-q_lower
    return SquaredIntervalExclusion(
        "numerically_excluded" if final_margin < 0 else "unresolved",
        lower, upper, beta_lower, q_lower,
        1 if final_margin < 0 else None, final_margin,
    )
