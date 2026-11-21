# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Published-geometry, deterministic Pareto-set sampling for DTLZ8.

The reference is a finite objective-space sample, not a canonical
probability distribution on the mixed-dimensional Pareto front.
"""
import numpy as np


def _component_counts(m, n_points):
    n_line = max(2, n_points // m)
    n_patch = n_points - n_line
    q, r = divmod(n_patch, m - 1)
    return n_line, [q + (k < r) for k in range(m - 1)]


def _hybrid_objectives(m, n_points):
    """Central line and M-1 structured surface patches (M>=3)."""
    n_line, counts = _component_counts(m, n_points)
    t = np.linspace(1.0 / 3.0, 1.0, n_line)
    line = np.empty((n_line, m))
    line[:, :-1] = ((1 - t) / 4)[:, None]
    line[:, -1] = t
    parts = [line]

    for k, count in enumerate(counts):
        # The first corner is unique to this patch. The common seam
        # is included once, on patch zero, when its budget permits.
        anchors = [(0.0, 0.0)]
        if k == 0 and count >= 2:
            anchors.append((0.0, 1.0))
        remaining = count - len(anchors)
        indices = np.arange(remaining)
        columns = max(1, int(np.ceil(np.sqrt(remaining))))
        rows = max(1, int(np.ceil(remaining / columns)))
        t_interior = (indices % columns + 0.5) / (3 * columns)
        u_interior = (indices // columns + 0.5) / rows
        t = np.concatenate(([a for a, _ in anchors], t_interior))
        u = np.concatenate(([b for _, b in anchors], u_interior))

        floor = (1 - t) / 4
        threshold = 1 - 2 * t
        low = floor + u * (threshold / 2 - floor)
        high = threshold - low
        patch = np.repeat(high[:, None], m, axis=1)
        patch[:, k] = low
        patch[:, -1] = t
        parts.append(patch)
    return np.vstack(parts)


def _legacy_m4_objectives(n_points):
    """Original M=4 on-Pareto dynamic grid, isolated for exact preservation."""
    m = 4
    points_per_arm = max(1, n_points // m)
    steps = int(np.sqrt(points_per_arm))
    if steps < 1:
        return np.empty((0, m))
    width = points_per_arm // steps
    fm = np.linspace(0.0, 1 / 3, steps, endpoint=False)
    fk_rel = np.linspace(0.0, 1.0, width)
    alpha = (1 - fm) / 4
    beta = 1 - 2 * fm
    fk = (alpha[:, None] + fk_rel[None, :] *
          (beta[:, None] / 2 - alpha[:, None])).ravel()
    fj = (beta[:, None] - (alpha[:, None] + fk_rel[None, :] *
          (beta[:, None] / 2 - alpha[:, None]))).ravel()
    tf = np.repeat(fm, width)
    tline = np.linspace(1 / 3, 1.0, points_per_arm)
    line = np.column_stack((
        np.repeat(((1 - tline) / 4)[:, None], m-1, axis=1), tline
    ))
    parts = [line]
    for k in range(m - 1):
        patch = np.empty((len(tf), m))
        patch[:, :-1] = fj[:, None]
        patch[:, k] = fk
        patch[:, -1] = tf
        parts.append(patch)
    return np.vstack(parts)


def _complete_m4(n_points):
    """Preserve all distinct legacy M=4 points; fill by farthest-first."""
    from scipy.spatial import cKDTree

    incumbent = _legacy_m4_objectives(n_points)
    _, positions = np.unique(incumbent, axis=0, return_index=True)
    incumbent = incumbent[np.sort(positions)]
    if len(incumbent) >= n_points:
        # For small budgets, start with the fully controlled hybrid.
        return _hybrid_objectives(4, n_points)

    lattice = np.unique(_hybrid_objectives(4, n_points), axis=0)
    distances = cKDTree(incumbent).query(lattice)[0] ** 2
    additions = []
    for _ in range(n_points - len(incumbent)):
        selected = int(np.argmax(distances))
        if distances[selected] <= 0:
            raise ValueError("Insufficient distinct DTLZ8 M=4 candidates.")
        point = lattice[selected]
        additions.append(point)
        squared = np.sum((lattice - point) ** 2, axis=1)
        distances = np.minimum(distances, squared)
        distances[selected] = -1
    return np.vstack((incumbent, np.asarray(additions)))


def dtlz8_objectives(m, n_points):
    """Generate exactly n_points unique Pareto objective vectors."""
    if not isinstance(n_points, (int, np.integer)):
        raise TypeError("n_points must be an integer.")
    if m < 2:
        raise ValueError("DTLZ8 requires M >= 2.")
    minimum = 2 if m == 2 else m + 1
    if n_points < minimum:
        raise ValueError(f"DTLZ8 requires n_points >= {minimum} for M={m}.")
    if m == 2:
        t = np.linspace(0, 1, n_points)
        objectives = np.column_stack(((1 - t) / 4, t))
    elif m == 4:
        objectives = _complete_m4(n_points)
    else:
        objectives = _hybrid_objectives(m, n_points)

    if len(objectives) != n_points:
        raise RuntimeError("DTLZ8 sampler failed requested cardinality.")
    if len(np.unique(objectives, axis=0)) != n_points:
        raise RuntimeError("DTLZ8 sampler generated duplicate objectives.")
    return objectives


def dtlz8_decisions(m, n_variables, n_points):
    """Invert each objective as the mean of an active decision block."""
    objectives = dtlz8_objectives(m, n_points)
    block_size = n_variables // m
    decisions = np.repeat(objectives, block_size, axis=1)
    remainder = n_variables % m
    if remainder:
        # DTLZ8 evaluation ignores these trailing variables.
        decisions = np.pad(decisions, ((0, 0), (0, remainder)),
                           mode="constant", constant_values=0.5)
    return decisions
