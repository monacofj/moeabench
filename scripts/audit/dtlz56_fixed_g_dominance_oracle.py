# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact (up to floating-point tolerance) dominance feasibility at a fixed g.

For original DTLZ5/DTLZ6, all objective vectors at a specified g lie
on a radius-(1+g) positive spherical patch with explicit angle bounds.
The greedy recursion is necessary and sufficient for existence of a
feasible point weakly no worse than a given nonnegative objective vector.

This is a research diagnostic. Checking a *single* g is not a global
Pareto optimality certificate; g must range continuously over its domain.
"""

from __future__ import annotations

import math

import numpy as np


def can_weakly_dominate_at_g(y, g: float, atol: float = 1e-12) -> bool:
    """Return whether a DTLZ5/6 solution at this g weakly dominates y.

    The feasible angle theta_1 is in [0,pi/2]. For each remaining angle
    theta_i (i>=2), lower=pi/[4(1+g)] and
    upper=pi*(1+2*g)/[4(1+g)].

    Allocate the largest permissible objective contribution to the last
    coordinate first, recursively; this minimizes residual radius while
    preserving every coordinate upper bound.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 2 or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite one-dimensional objective vector")
    if g < 0 or not np.isfinite(g):
        raise ValueError("g must be nonnegative and finite")
    if np.any(y < -atol):
        return False

    radius = 1.0 + float(g)
    if y[-1] >= radius - atol:
        return True
    if y[-1] < -atol:
        return False

    remaining = math.sqrt(max(0.0, radius**2 - max(0.0, y[-1])**2))
    lo = math.pi / (4.0*(1+g))
    hi = math.pi * (1+2*g) / (4.0*(1+g))

    for j in range(len(y)-2, 0, -1):
        if remaining * math.sin(lo) > y[j] + atol:
            return False
        available = min(max(0.0, y[j]), remaining * math.sin(hi))
        remaining = math.sqrt(max(0.0, remaining**2-available**2))

    return remaining <= y[0] + atol
