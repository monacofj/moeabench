# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent published DPF3/DPF5 equation controls; issue #119.

Tests exercise the production MOP evaluators but construct expected values
from separate reference formulas, not the tested private kernel.
No pre-existing regression test or frozen numerical fixture is changed.
"""

import numpy as np
import pytest

from moeabench.mops.DPF3 import DPF3
from moeabench.mops.DPF5 import DPF5


def _sphere(angles):
    n, q = angles.shape
    values = np.empty((n, q + 1))
    for i in range(q + 1):
        for row in range(n):
            value = 1.0
            for j in range(q - i):
                value *= float(np.cos(angles[row, j]))
            if i:
                value *= float(np.sin(angles[row, q - i]))
            values[row, i] = value
    return values


def _dpf3_reference(X, m, d, eta):
    theta = (np.pi / 2) * X[:, :d-1]**100
    h = _sphere(theta)
    g = np.sum((X[:, d-1:] - 0.5)**2, axis=1)
    gamma = (1 + g[:, None]) * (1 - h)
    t = gamma[:, -1]
    projections = [np.minimum(t, eta[0])]
    for a, b in zip(eta[:-1], eta[1:]):
        projections.append(np.minimum(np.maximum(t, a), b))
    projections.append(np.maximum(t, eta[-1]))
    return np.column_stack([gamma[:, :-1], np.column_stack(projections)])


def _dpf5_reference(X, m, d):
    theta = X[:, :m-1] * (np.pi / 2)
    P = _sphere(theta)
    h = _sphere(theta[:, :d-1])
    n_repeats = m - d + 1
    D = np.column_stack([
        np.repeat(h[:, :1] / np.sqrt(n_repeats), n_repeats, axis=1),
        h[:, 1:],
    ])
    g = (X[:, m-1] - X[:, 0])**2
    g += np.sum((X[:, m:] - 0.5)**2, axis=1)
    branch_D = X[:, 0] >= 1 / 3
    return np.where(branch_D[:, None], D, P) * (1 + g[:, None])


@pytest.mark.parametrize("m,d", [(3,2),(5,2),(6,3),(7,4)])
def test_dpf3_matches_independent_sorted_published_projection(m, d):
    mop = DPF3(M=m, D=d)
    rng = np.random.default_rng(m * 31 + d)
    X = rng.uniform(0.68, 0.97, size=(80, mop.N))
    X[:, d-1:] = rng.uniform(0.1, 0.9, size=(80, mop.N - d + 1))
    eta = np.sort(mop._chaos_pool[:m-d])
    current = mop.evaluation(X)["F"]
    expected = _dpf3_reference(X, m, d, eta)
    np.testing.assert_allclose(current, expected, rtol=0, atol=2e-13)
    projected = current[:, d-1:]
    t = (1 + np.sum((X[:, d-1:] - 0.5)**2, axis=1)) * (1 - _sphere(X[:, :d-1]**100 * np.pi/2)[:, -1])
    np.testing.assert_allclose(projected.sum(axis=1), t + eta.sum(), rtol=0, atol=2e-13)


def test_dpf3_first_objective_does_not_collapse_at_small_positive_angles():
    mop = DPF3(M=3, D=2)
    X = np.full((1, mop.N), 0.5)
    X[0, 0] = 0.7
    result = mop.evaluation(X)["F"][0, 0]
    assert result > 0
    angle = (0.7**100) * np.pi/2
    assert 1 - np.cos(angle) == 0.0
    np.testing.assert_allclose(result, 2 * np.sin(angle/2)**2, rtol=1e-12, atol=0)


@pytest.mark.parametrize("m,d", [(3,2),(4,2),(5,2),(6,3),(6,4),(8,4)])
def test_dpf5_matches_independent_published_formulas_and_radial_norm(m, d):
    mop = DPF5(M=m, D=d)
    rng = np.random.default_rng(m * 73 + d)
    X = rng.uniform(0.05, 0.95, size=(125, mop.N))
    X[0, 0] = 1 / 3
    X[1, 0] = np.nextafter(1 / 3, 0.0)
    X[2, 0] = np.nextafter(1 / 3, 1.0)
    current = mop.evaluation(X)["F"]
    expected = _dpf5_reference(X, m, d)
    np.testing.assert_allclose(current, expected, rtol=0, atol=3e-14)
    g = (X[:, m-1] - X[:, 0])**2 + np.sum((X[:, m:] - 0.5)**2, axis=1)
    np.testing.assert_allclose(np.linalg.norm(current, axis=1), 1 + g, rtol=0, atol=4e-14)


def test_dpf5_inclusive_boundary_belongs_to_degenerate_branch():
    m, d = 5, 2
    mop = DPF5(M=m, D=d)
    X = np.full((1, mop.N), 0.5)
    X[0, 0] = 1 / 3
    X[0, m-1] = X[0, 0]
    values = mop.evaluation(X)["F"][0]
    h1 = np.cos(np.pi/6)
    np.testing.assert_allclose(values[:m-d+1], np.full(m-d+1, h1 / np.sqrt(m-d+1)), rtol=0, atol=2e-14)
    assert values[-1] == pytest.approx(np.sin(np.pi/6), abs=2e-14)


def test_dpf5_g_zero_decisions_are_globally_nondominated_by_radial_witness():
    mop = DPF5(M=6, D=3)
    rng = np.random.default_rng(19)
    X = rng.random((96, mop.N))
    X[:, mop.M-1] = X[:, 0]
    X[:, mop.M:] = 0.5
    F0 = mop.evaluation(X)["F"]
    np.testing.assert_allclose(np.linalg.norm(F0, axis=1), 1.0, rtol=0, atol=3e-14)
    off = X.copy()
    off[:, mop.M-1] = np.clip(off[:, mop.M-1] + .05, 0, 1)
    F1 = mop.evaluation(off)["F"]
    assert np.all(F1 >= F0 - 3e-14)
    assert np.all(np.any(F1 > F0 + 1e-8, axis=1))
