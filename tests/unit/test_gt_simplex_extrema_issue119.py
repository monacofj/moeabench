# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent positive checks for issue #119 simplex extreme coverage."""
import numpy as np
import pytest

from moeabench.mops.DTLZ1 import DTLZ1
from moeabench.mops.DPF1 import DPF1
from moeabench.mops._simplex_anchors import (
    anchor_simplex_positions, simplex_vertex_positions,
)


@pytest.mark.parametrize("dimension", [2, 3, 4, 5, 9])
def test_simplex_decisions_evaluate_to_exact_essential_vertices(dimension):
    vertices = simplex_vertex_positions(dimension)
    mop = DTLZ1(M=dimension)
    decisions = np.full((dimension, mop.N), 0.5)
    decisions[:, :dimension-1] = vertices
    actual = mop.evaluation(decisions)["F"]
    np.testing.assert_allclose(actual, 0.5*np.eye(dimension),
                               rtol=0, atol=1e-14)


@pytest.mark.parametrize("dimension,budget", [(2, 2), (3, 128), (5, 512),
                                               (9, 128), (5, 3)])
def test_dtlz1_retains_vertex_coverage_and_unchanged_random_tail(dimension, budget):
    mop = DTLZ1(M=dimension)
    decisions = mop.ps(n_points=budget)
    assert decisions.shape == (budget, mop.N)
    assert np.array_equal(decisions, mop.ps(n_points=budget))
    first = min(budget, dimension)
    np.testing.assert_array_equal(decisions[:first, :dimension-1],
                                  simplex_vertex_positions(dimension)[:first])
    if budget >= dimension:
        np.testing.assert_allclose(mop.evaluation(decisions)["F"][:dimension],
                                   0.5 * np.eye(dimension), rtol=0, atol=1e-14)
    random_original = np.random.RandomState(42).random((budget, dimension - 1))
    np.testing.assert_array_equal(decisions[first:, :dimension-1],
                                  random_original[first:])
    np.testing.assert_array_equal(decisions[:, dimension-1:], 0.5)


@pytest.mark.parametrize("m,d,budget", [(5, 2, 128), (5, 3, 128),
                                         (7, 4, 512), (10, 6, 128)])
def test_dpf1_preserves_extreme_essential_simplex_under_projection(m, d, budget):
    mop = DPF1(M=m, D=d)
    decisions = mop.ps(n_points=budget)
    values = mop.evaluation(decisions)["F"]
    assert values.shape == (budget, m)
    assert np.array_equal(decisions, mop.ps(n_points=budget))
    if d == 2:
        # The incumbent D=2 structured grid is already endpoint-inclusive.
        np.testing.assert_array_equal(decisions[:, 0],
                                      np.linspace(0, 1, budget))
        extrema = values[[0, -1], :d]
        np.testing.assert_allclose(extrema,
                                   np.array([[0.0, 0.5], [0.5, 0.0]]),
                                   atol=1e-14, rtol=0)
    else:
        np.testing.assert_allclose(values[:d, :d], 0.5 * np.eye(d),
                                   atol=1e-14, rtol=0)
        np.testing.assert_array_equal(decisions[:d, :d-1],
                                      simplex_vertex_positions(d))
        rng_rows = np.random.RandomState(42).random((budget, d-1))
        np.testing.assert_array_equal(decisions[d:, :d-1], rng_rows[d:])


def test_vertex_anchor_count_does_not_exceed_requested_budget():
    positions = np.full((2, 4), 0.4)
    anchor_simplex_positions(positions, 5)
    assert positions.shape == (2, 4)
    np.testing.assert_array_equal(positions,
                                  simplex_vertex_positions(5)[:2])
