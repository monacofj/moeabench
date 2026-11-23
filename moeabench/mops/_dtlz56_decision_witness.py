# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent decision-to-objective witness verification for DTLZ5/6.

Uses published equations directly; never calls production evaluation,
pf, ps, or the scalarizer. A consistent feasible decision witness
does not prove global Pareto efficiency of its objectives.
"""
from __future__ import annotations
import numpy as np


def inspect_dtlz56_decision_witness(
    decisions, objectives, *, problem, m, tolerance=1e-9
):
    """Verify one-to-one feasible decision and declared objective rows."""
    x = np.asarray(decisions, dtype=float)
    f = np.asarray(objectives, dtype=float)
    if problem not in ("DTLZ5", "DTLZ6") or m < 2:
        raise ValueError("Expected original DTLZ5/6 and M >= 2")
    if x.ndim != 2 or x.shape[1] < m or len(x) == 0:
        raise ValueError("Decision reference must have shape (n,N), N >= M")
    if f.ndim != 2 or f.shape != (len(x), m):
        raise ValueError("Decision and GT rows must correspond exactly")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    finite_rows = np.isfinite(x).all(axis=1) & np.isfinite(f).all(axis=1)
    bounds_rows = finite_rows & (x >= -tolerance).all(axis=1) & (
        x <= 1+tolerance).all(axis=1)
    outside = int(np.count_nonzero(~bounds_rows))
    if outside:
        return {
            "violations": outside, "invalid_decisions": outside,
            "objective_mismatch": 0, "max_objective_residual": None,
            "decision_rows": int(len(x)),
        }

    z = np.clip(x, 0.0, 1.0)
    tail = z[:, m-1:]
    if problem == "DTLZ5":
        g = np.sum((tail-.5)**2, axis=1)
    else:
        g = np.sum(tail**.1, axis=1)

    theta = np.empty((len(x), m-1), dtype=float)
    theta[:, 0] = np.pi*z[:, 0]/2
    for i in range(1, m-1):
        theta[:, i] = np.pi*(1+2*g*z[:, i])/(4*(1+g))

    predicted = np.empty((len(x), m), dtype=float)
    for j in range(m):
        expected = 1+g
        if j < m-1:
            expected = expected*np.prod(np.cos(theta[:, :m-j-1]), axis=1)
        if j > 0:
            expected = expected*np.sin(theta[:, m-j-1])
        predicted[:, j] = expected

    residual = np.max(np.abs(predicted-f), axis=1)
    mismatch = int(np.count_nonzero(residual > tolerance))
    return {
        "violations": mismatch, "invalid_decisions": 0,
        "objective_mismatch": mismatch,
        "max_objective_residual": float(np.max(residual)),
        "decision_rows": int(len(x)),
    }
