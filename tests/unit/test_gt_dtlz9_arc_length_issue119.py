# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent DTLZ9 published-curve and arc-length GT contracts (#119)."""
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ellipeinc

from moeabench.mops.DTLZ9 import DTLZ9
from moeabench.mops._dtlz9_arc import (
    dtlz9_equal_arc_angles, dtlz9_equal_arc_objectives,
)


@pytest.mark.parametrize("m,budget", [
    (2, 128), (3, 128), (5, 512), (10, 128),
    (3, 2000), (10, 2000),
])
def test_dtlz9_new_ps_exact_pareto_curve_and_replay(m, budget):
    mop = DTLZ9(M=m)
    x = mop.ps(n_points=budget)
    f = mop.evaluation(x, n_ieq_constr=1)["F"]
    assert x.shape == (budget, mop.N)
    assert len(np.unique(f, axis=0)) == budget
    np.testing.assert_array_equal(x, mop.ps(n_points=budget))
    np.testing.assert_allclose(f[:, :-1],
                               np.repeat(f[:, :1], m-1, axis=1),
                               atol=2e-14, rtol=0)
    residual = f[:, :-1]**2 + f[:, -1:]**2 - 1
    assert np.max(np.abs(residual)) <= 2e-14
    assert np.isclose(f[0, -1], 0, atol=1e-15)
    assert np.isclose(f[-1, -1], 1, atol=1e-15)


@pytest.mark.parametrize("m,budget", [(2, 128), (3, 128), (5, 512), (10, 2000)])
def test_dtlz9_arcs_equal_by_independent_quadrature(m, budget):
    f = dtlz9_equal_arc_objectives(m, budget)
    theta = np.arctan2(f[:, -1], f[:, 0])
    # This verifies the defining arc-length integral, independently
    # of the helper's elliptic Newton iteration.
    knots = np.linspace(0, budget-1, min(9, budget), dtype=int)
    lengths = [
        quad(lambda t: np.sqrt((m-1)*np.sin(t)**2 + np.cos(t)**2),
             0, float(theta[k]), epsabs=1e-12, epsrel=1e-12)[0]
        for k in knots
    ]
    total = lengths[-1]
    np.testing.assert_allclose(lengths,
                               total * knots/(budget-1),
                               atol=3e-12, rtol=0)
    if m > 2:
        original_theta = np.arcsin(np.linspace(0, 1, budget))
        old_segments = np.diff(
            ellipeinc(original_theta, 2-m))
        new_segments = np.diff(
            ellipeinc(theta, 2-m))
        assert np.std(new_segments) < 1e-10 * np.mean(new_segments)
        assert np.std(old_segments) > 0.01 * np.mean(old_segments)


def test_dtlz9_equal_arc_angles_small_budget_and_validation():
    np.testing.assert_array_equal(dtlz9_equal_arc_angles(3, 1), [0.0])
    with pytest.raises(ValueError, match="positive integer"):
        dtlz9_equal_arc_angles(3, 0)
    with pytest.raises(ValueError, match="M>=2"):
        dtlz9_equal_arc_angles(1, 128)


def test_dtlz9_preserves_nonmultiple_N_tail_behavior():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mop = DTLZ9(M=3, N=31)
    assert any("ignored" in str(item.message) for item in w)
    decision = mop.ps(128)
    assert decision.shape == (128, 31)
    assert np.all(decision[:, -1] == 0)
    f = mop.evaluation(decision)["F"]
    np.testing.assert_allclose(f[:, 0], f[:, 1], atol=1e-14, rtol=0)
