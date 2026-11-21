# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact position-space anchors for DTLZ1-like essential simplex vertices.

The published DTLZ1 position mapping has D axis vertices. This helper
changes only position samples; the MOP owns distance variables and
projection onto its complete objective space.
"""
import numpy as np


def simplex_vertex_positions(d):
    """Return one distinct decision-position row for each essential vertex."""
    if not isinstance(d, (int, np.integer)) or d < 2:
        raise ValueError("Simplex objective dimension must be at least 2.")
    vertices = np.ones((d, d - 1), dtype=float)
    for objective in range(1, d):
        # At the first zero of the DTLZ1 product, only f_objective survives.
        vertices[objective, d - objective - 1] = 0.0
    return vertices


def anchor_simplex_positions(positions, d):
    """Replace the first min(n,D) decision positions by exact vertices.

    The function never changes count, random source or remaining rows.
    For n<D complete coverage is mathematically impossible and no
    assertion of full coverage is made.
    """
    if positions.ndim != 2 or positions.shape[1] != d - 1:
        raise ValueError("Position array must have D-1 columns.")
    n = min(len(positions), d)
    positions[:n] = simplex_vertex_positions(d)[:n]
    return positions
