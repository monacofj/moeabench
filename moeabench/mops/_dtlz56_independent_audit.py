# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent necessary-geometry and exact g=0 membership for DTLZ5/6.

This module does NOT import or call the production scalarizer, Pareto
sampler, objective evaluator, or the DTLZ2 angle inversion helper.
For M >= 4, feasible minimal-radius g>0 boundary points can still
dominate one another. Those points are explicitly unverified until
independent GLOBAL Pareto membership is established.
"""
from __future__ import annotations

import numpy as np


def inspect_dtlz56_geometry(points, *, m, k, problem, tolerance=1e-9):
    """Return independent published-equation diagnostics for objective points.

    Pareto necessity: r=1+g and every inner angle theta_i lies in the
    original g-dependent angular box. At fixed objective direction
    the minimum attainable radius occurs at
        q = max_i abs(4*theta_i/pi - 1)
        g_min = q/(1-q).
    A larger g is STRICTLY dominated by a feasible lower-radius point
    on the same objective ray.

    Sufficiency for g=0: every feasible objective has Euclidean norm
    at least 1, so no other feasible objective can dominate a unit-
    norm g=0 point unless it is identical.
    """
    if problem not in ("DTLZ5", "DTLZ6") or m < 2 or k < 1:
        raise ValueError("Expected original DTLZ5/6 dimensions")
    f = np.asarray(points, dtype=float)
    if f.ndim != 2 or f.shape[1] != m or not np.isfinite(f).all():
        raise ValueError("Expected finite objective vectors of shape (n, M)")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")

    radius = np.linalg.norm(f, axis=1)
    g = radius - 1.0
    g_max = k/4.0 if problem == "DTLZ5" else float(k)
    inner = []
    for i in range(1, m-1):
        index = m-i-1
        theta = np.arctan2(f[:, index],
                           np.linalg.norm(f[:, :index], axis=1))
        inner.append(theta)
    if inner:
        angles = np.column_stack(inner)
        q = np.max(np.abs(4.0*angles/np.pi - 1.0), axis=1)
    else:
        q = np.zeros(len(f))
    g_min = np.divide(q, 1-q, out=np.full_like(q, np.inf), where=q < 1)

    nonnegative = np.all(f >= -tolerance, axis=1)
    feasible_radius = (g >= -tolerance) & (g <= g_max+tolerance)
    feasible_angles = (q <= g/np.maximum(1+g, 1e-14) + tolerance)
    feasible = nonnegative & feasible_radius & feasible_angles
    radially_dominated = feasible & (g > g_min + tolerance)
    certified_g0 = feasible & (np.abs(g) <= tolerance)
    unresolved = feasible & ~radially_dominated & ~certified_g0
    bad = ~feasible | radially_dominated
    max_radial_excess = float(np.max(np.maximum(g-g_min, 0.0))) if len(g) else 0.0
    return {
        "violations": int(np.count_nonzero(bad)),
        "outside_feasible_image": int(np.count_nonzero(~feasible)),
        "strictly_radially_dominated": int(np.count_nonzero(radially_dominated)),
        "certified_g0": int(np.count_nonzero(certified_g0)),
        "unverified_positive_g": int(np.count_nonzero(unresolved)),
        "g_positive": int(np.count_nonzero(feasible & (g > tolerance))),
        "g_zero": int(np.count_nonzero(certified_g0)),
        "max_radial_excess": max_radial_excess,
        "g_max": float(g_max),
    }
