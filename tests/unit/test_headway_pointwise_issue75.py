# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pointwise Headway classifications must use real P0, never frozen rand50."""

import numpy as np
import pytest

from moeabench.diagnostics import baselines, qscore


def _front():
    x = np.linspace(0.0, 1.0, 41)
    return np.column_stack((x, 1-x, np.zeros_like(x)))


def test_headway_pointwise_uses_actual_p0_gd95_and_analytic_q(monkeypatch):
    gt = _front()
    points = gt[::4]
    p0 = points + [0.0, 0.0, 0.4]
    pf = points + [0.0, 0.0, 0.1]

    def forbidden(*args, **kwargs):
        raise AssertionError("An old headway.rand50 distribution was consulted")

    monkeypatch.setattr(baselines, "get_baseline_values", forbidden)
    monkeypatch.setattr(baselines, "get_baseline_ecdf", forbidden)
    q = qscore.q_headway_points(pf, ref=gt, initial_data=p0, s_k=0.2)
    expected = 1.0 - np.log1p(0.25)/np.log(2.0)
    np.testing.assert_allclose(q, expected, atol=1e-12, rtol=0)
    assert np.all((q >= 0) & (q <= 1))


def test_headway_pointwise_without_true_p0_is_nan_not_random_baseline():
    gt = _front()
    pf = gt[::4] + [0.0, 0.0, 0.1]
    result = qscore.q_headway_points(pf, ref=gt, s_k=0.2)
    assert len(result) == len(pf)
    assert np.all(np.isnan(result))


def test_headway_pointwise_responds_to_heterogeneous_final_distances():
    gt = _front()
    p0 = gt[::4] + [0.0, 0.0, 0.4]
    pf = gt[::4].copy()
    pf[:, 2] = np.linspace(0.0, 0.8, len(pf))
    result = qscore.q_headway_points(pf, ref=gt, initial_data=p0, s_k=500.0)
    # s_K intentionally irrelevant to the dimensionless longitudinal ratio.
    ratios = pf[:, 2]/0.4
    expected = 1-np.clip(np.log1p(ratios)/np.log(2.0), 0, 1)
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0)
    assert result[0] == pytest.approx(1.0)
    assert result[-1] == pytest.approx(0.0)


def test_headway_pointwise_with_zero_error_initial_state():
    gt = _front()
    p0 = gt[::4]
    pf = p0.copy()
    pf[1, 2] = 0.01
    result = qscore.q_headway_points(pf, ref=gt, initial_data=p0)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)
