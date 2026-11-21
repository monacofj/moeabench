# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent DPF4 publication-equation regression tests (issue #119)."""
import numpy as np
import pytest

from moeabench.mops.DPF4 import DPF4


def published_objectives(x, mop):
    """Independent equation in objective space, with ordered eta thresholds."""
    d, m = mop.D, mop.M
    angles = np.pi * x[:, :d - 1] / 2
    h = np.ones((len(x), d))
    for j in range(d):
        for k in range(d - j - 1):
            h[:, j] *= np.cos(angles[:, k])
        if j:
            h[:, j] *= np.sin(angles[:, d - j - 1])
    tail = x[:, d - 1:] - 0.5
    g = 100 * (tail.shape[1] + np.sum(
        tail**2 - np.cos(20 * np.pi * tail), axis=1))
    base = (1 + g[:, None]) * (1 - h)
    eta = np.sort(mop._chaos_pool[:m - d])
    t = base[:, -1]
    projection = [np.minimum(t, eta[0])]
    for lower, upper in zip(eta[:-1], eta[1:]):
        projection.append(np.minimum(np.maximum(t, lower), upper))
    projection.append(np.maximum(t, eta[-1]))
    return np.column_stack((base[:, :-1], *projection)) ** 2


@pytest.mark.parametrize("m,d", [(3, 2), (5, 2), (6, 3), (7, 4)])
def test_dpf4_matches_published_sorted_projection(m, d):
    mop = DPF4(M=m, D=d)
    rng = np.random.default_rng(11942)
    x = rng.uniform(0.05, 0.95, size=(256, mop.N))
    # Include both Pareto-set points (g=0) and off-front decisions (g>0).\n    x[:128, d - 1:] = 0.5\n    np.testing.assert_allclose(
        mop.evaluation(x)["F"], published_objectives(x, mop),
        rtol=2e-12, atol=2e-13)


def test_dpf4_small_angle_remains_positive():
    mop = DPF4(M=3, D=2)
    x = np.full((1, mop.N), 0.5)
    x[0, 0] = 1e-10
    value = mop.evaluation(x)["F"][0, 0]
    angle = np.pi * x[0, 0] / 2
    expected = (2 * np.sin(angle / 2)**2)**2
    assert value > 0
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=0)


def test_dpf4_does_not_mutate_shared_chaos_pool():
    mop = DPF4(M=7, D=2)
    before = mop._chaos_pool.copy()
    x = np.full((4, mop.N), 0.5)
    mop.evaluation(x)
    np.testing.assert_array_equal(mop._chaos_pool, before)
