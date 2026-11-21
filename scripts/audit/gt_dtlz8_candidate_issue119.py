# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only DTLZ8 finite-sample candidate; not a production GT policy.

For M>=3, the analytic nondominated front is the union of a central
line (t in [1/3,1]) and M-1 two-dimensional patches (t in [0,1/3]).
The patch parameterization is obtained by making the smallest pair
sum equal 1-2t while respecting each individual constraint.
This module does not claim a canonical mixed-dimensional measure.
"""
import numpy as np


def _radical_inverse(index, base):
    result, fraction = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * fraction
        fraction /= base
    return result


def candidate_objectives(m, n_points, *, patch_strategy="halton"):
    """Produce exactly n distinct feasible PF objective vectors.

    The allocation is explicitly a coverage policy (one component
    per region), not a probability measure on the mixed-dimensional PF.
    """
    if patch_strategy not in ("halton", "lattice", "hybrid"):
        raise ValueError("Unknown patch strategy.")
    if m == 2:
        if n_points < 2:
            raise ValueError("M=2 requires at least two line endpoints.")
        t = np.linspace(0.0, 1.0, n_points)
        return np.column_stack(((1.0 - t) / 4, t))
    if m < 2:
        raise ValueError("DTLZ8 requires M>=2.")
    if n_points < m + 1:
        raise ValueError("Need at least M+1 points to sample all regions and line endpoints.")

    # A lattice variant assigns slightly more rows to two-dimensional
    # patches; the line remains explicitly anchored at both endpoints.
    line_divisor = m + 1 if patch_strategy == "lattice" else m
    n_line = max(2, n_points // line_divisor)
    n_patch = n_points - n_line
    q, r = divmod(n_patch, m - 1)
    counts = [q + (i < r) for i in range(m - 1)]
    if min(counts) < 1:
        raise ValueError("Requested budget cannot cover all patches.")

    # One-dimensional line: all first M-1 objectives equal (1-t)/4.
    t = np.linspace(1.0 / 3, 1.0, n_line)
    line = np.empty((n_line, m), dtype=float)
    line[:, :-1] = ((1.0 - t) / 4)[:, None]
    line[:, -1] = t
    parts = [line]

    # For t<1/3: the two smallest first objectives sum to 1-2t.
    # Exactly one is the 'low' coordinate, and all others are 'high'.
    for low_index, count in enumerate(counts):
        # Include the t=0 low/high extreme for each patch. The common
        # t=0 equal-coordinate edge is inserted once, on patch zero.
        anchors = [(0.0, 0.0)]
        if low_index == 0 and count >= 2:
            anchors.append((0.0, 1.0))
        remaining = count - len(anchors)
        k = np.arange(remaining)
        if patch_strategy == "halton":
            t_interior = np.array([_radical_inverse(int(i) + 1, 2)
                                   for i in k]) / 3
            u_interior = np.array([_radical_inverse(int(i) + 1, 3)
                                   for i in k])
        else:
            columns = max(1, int(np.ceil(np.sqrt(remaining))))
            rows = max(1, int(np.ceil(remaining / columns)))
            t_interior = (k % columns + 0.5) / (3 * columns)
            u_interior = (k // columns + 0.5) / rows
        t = np.concatenate(([p[0] for p in anchors], t_interior))
        u = np.concatenate(([p[1] for p in anchors], u_interior))
        floor = (1.0 - t) / 4
        threshold = 1.0 - 2.0 * t
        low = floor + u * (threshold / 2 - floor)
        high = threshold - low
        patch = np.repeat(high[:, None], m, axis=1)
        patch[:, low_index] = low
        patch[:, -1] = t
        parts.append(patch)
    return np.vstack(parts)


def candidate_decisions(m, n_points, n_variables=None, *, patch_strategy="halton"):
    """Map objective vectors to constant-block feasible decisions."""
    if n_variables is None:
        n_variables = 10 * m
    if n_variables < m:
        raise ValueError("N must be at least M for DTLZ8.")
    objectives = candidate_objectives(m, n_points, patch_strategy=patch_strategy)
    active = np.repeat(objectives, n_variables // m, axis=1)
    # Production DTLZ8 currently ignores N % M trailing variables.
    # Keep that public contract for an explicit non-multiple N.
    ignored = n_variables % m
    if ignored:
        active = np.pad(active, ((0, 0), (0, ignored)),
                        mode="constant", constant_values=0.5)
    return active
