# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent published-equation checks of production DTLZ8 PS/PF.

These assertions replace invalid finite-reference assumptions with
analytical feasibility, Pareto membership, coverage and API contracts.
"""
from itertools import combinations
import warnings

import numpy as np
import pytest

from moeabench.mops.DTLZ8 import DTLZ8


def _published_residual(f):
    m = f.shape[1]
    t = f[:, -1]
    y = f[:, :-1]
    if m == 2:
        return np.abs(y[:, 0] - (1 - t) / 4)
    r = np.zeros(len(f))
    line = t >= 1 / 3
    if np.any(line):
        r[line] = np.max(
            np.abs(y[line] - ((1 - t[line]) / 4)[:, None]), axis=1)
    if np.any(~line):
        z = np.sort(y[~line], axis=1)
        threshold = 1 - 2 * t[~line]
        floor = (1 - t[~line]) / 4
        pair = np.abs(z[:, 0] + z[:, 1] - threshold)
        floor_error = np.maximum(floor - z[:, 0], 0)
        upper_equal = (np.max(np.abs(z[:, 2:] - z[:, 1:2]), axis=1)
                       if m > 3 else np.zeros(len(z)))
        r[~line] = np.maximum.reduce(
            (pair, floor_error, upper_equal))
    return r


@pytest.mark.parametrize("m", [2, 3, 4, 5, 6, 10])
@pytest.mark.parametrize("count", [100, 128, 512, 2000])
def test_dtlz8_production_samples_published_pf_exact_budget(m, count):
    mop = DTLZ8(M=m)
    x = mop.ps(n_points=count)
    f = mop.pf(n_points=count)
    assert x.shape == (count, mop.N)
    assert f.shape == (count, m)
    assert np.unique(f, axis=0).shape[0] == count
    assert np.max(_published_residual(f)) <= 2e-13
    np.testing.assert_allclose(mop.evaluation(x)["F"], f, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(mop.ps(n_points=count), x)
    t = f[:, -1]
    assert np.all(t[:, None] + 4 * f[:, :-1] >= 1 - 2e-13)
    if m >= 3:
        pairs = np.stack([f[:, i] + f[:, j]
                          for i, j in combinations(range(m - 1), 2)])
        assert np.all(2*t + pairs.min(axis=0) >= 1 - 2e-13)
        assert np.any(t >= 1 / 3)
        for k in range(m-1):
            subset = f[t < 1 / 3]
            assert np.any(np.argmin(subset[:, :-1], axis=1) == k)


@pytest.mark.parametrize("m,n", [(2, 21), (3, 32), (4, 43), (5, 52), (10, 103)])
def test_dtlz8_nonmultiple_n_preserves_constructor_and_mapping(m, n):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        mop = DTLZ8(M=m, N=n)
    assert any("multiple" in str(r.message) for r in recorded)
    x = mop.ps(n_points=128)
    assert x.shape == (128, n)
    assert np.all(x[:, (n//m)*m:] == 0.5)
    assert np.max(_published_residual(mop.evaluation(x)["F"])) < 1e-13


@pytest.mark.parametrize("m", [2, 3, 4, 10])
def test_dtlz8_rejects_undercovered_budgets(m):
    mop = DTLZ8(M=m)
    minimum = 2 if m == 2 else m + 1
    with pytest.raises(ValueError, match="n_points"):
        mop.ps(n_points=minimum - 1)


def test_dtlz8_default_public_sampling_budget_unchanged():
    mop = DTLZ8(M=3)
    assert mop.ps().shape == (100, mop.N)
    assert mop.pf().shape == (100, mop.M)
