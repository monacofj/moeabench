# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analytical reduction of the original DTLZ5/DTLZ6 complete-front search.

At any fixed objective direction, the smallest feasible g must be used
on the global Pareto front. This does NOT certify global Pareto membership.
"""

import numpy as np
import pytest

from moeabench.mops.DTLZ5 import DTLZ5
from moeabench.mops.DTLZ6 import DTLZ6


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_nonminimal_g_is_strictly_dominated_at_fixed_objective_direction(cls):
    """Independent proof witness: reduce g without changing any angles."""
    mop = cls(M=4)
    k = mop.N - mop.M + 1
    g = 0.328135
    u = np.array([0.976618, 0.066887])
    x = np.full((1, mop.N), 0.5)
    x[0, :3] = [0.173406, *u]
    x[0, 3:] = (
        0.5 + np.sqrt(g/k) if cls is DTLZ5 else (g/k)**10
    )
    original = mop.evaluation(x)["F"][0]

    # theta_i - pi/4 = (pi/4) * [g/(1+g)] * (2*u_i - 1).
    q = g / (1+g) * np.max(np.abs(2*u - 1))
    g_min = q / (1-q)
    assert 0 < g_min < g

    u_min = ((1+g_min) * (1+2*g*u)/(1+g) - 1) / (2*g_min)
    assert np.all(u_min >= -1e-12)
    assert np.all(u_min <= 1+1e-12)
    assert np.any(np.isclose(u_min, 0) | np.isclose(u_min, 1))

    x_min = x.copy()
    x_min[0, 1:3] = u_min
    x_min[0, 3:] = (
        0.5 + np.sqrt(g_min/k) if cls is DTLZ5 else (g_min/k)**10
    )
    reduced = mop.evaluation(x_min)["F"][0]
    np.testing.assert_allclose(
        reduced, original * (1+g_min)/(1+g), atol=5e-13, rtol=0
    )
    assert np.all(reduced < original)


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_g_zero_curve_cannot_be_extended_by_nonminimal_radius(cls):
    """The canonical g=0 direction already attains the minimal norm."""
    mop = cls(M=4)
    x = np.full((1, mop.N), 0.5)
    x[0, 0] = 0.25
    x[0, 3:] = 0.5 if cls is DTLZ5 else 0.0
    g_zero = mop.evaluation(x)["F"][0]
    np.testing.assert_allclose(np.linalg.norm(g_zero), 1.0, atol=1e-14)
