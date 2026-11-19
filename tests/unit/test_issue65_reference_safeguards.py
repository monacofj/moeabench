# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused safeguards for known high-dimensional analytical-reference limitations."""

import numpy as np
import pytest

from moeabench.mops import DTLZ2, DTLZ5, DTLZ6, DPF1, DPF5


@pytest.mark.parametrize("mop_cls", [DTLZ5, DTLZ6])
@pytest.mark.parametrize("M", [4, 5, 10])
def test_dtlz56_warns_that_g_zero_is_an_incomplete_front(mop_cls, M):
    mop = mop_cls(M=M)
    with pytest.warns(UserWarning, match="not a complete Ground Truth"):
        ps = mop.ps(n_points=32)
    assert ps.shape == (32, mop.N)
    with pytest.warns(UserWarning, match="not a complete Ground Truth"):
        front = mop.pf(n_points=32)
    assert front.shape == (32, M)
    np.testing.assert_allclose(np.sum(front**2, axis=1), 1.0, atol=1e-12, rtol=0)


@pytest.mark.parametrize("mop_cls", [DTLZ5, DTLZ6])
@pytest.mark.parametrize("M", [2, 3])
def test_dtlz56_valid_legacy_front_has_no_incompleteness_warning(mop_cls, M):
    mop = mop_cls(M=M)
    front = mop.pf(n_points=32)
    assert front.shape == (32, M)
    np.testing.assert_allclose(np.sum(front**2, axis=1), 1.0, atol=1e-12, rtol=0)


@pytest.mark.parametrize("M,N", [(7,6), (10,6), (12,8)])
def test_dpf5_rejects_underspecified_position_variables_before_math(M, N):
    mop = DPF5(M=M, N=N, D=2, K=5)
    assert mop.N == N  # Keep existing BaseDPF construction contract.
    with pytest.raises(ValueError, match=r"DPF5 requires N >= M"):
        mop.evaluation(np.full((2, N), 0.5))
    with pytest.raises(ValueError, match=r"Pass N >="):
        mop.ps(n_points=16)


@pytest.mark.parametrize("M", [7, 10])
def test_dpf5_works_with_explicit_sufficient_decision_variables(M):
    mop = DPF5(M=M, N=M+2, D=2, K=5)
    ps = mop.ps(n_points=16)
    front = mop.evaluation(ps)["F"]
    assert ps.shape == (16, M+2)
    assert front.shape == (16, M)
    assert np.isfinite(front).all()


def test_other_dpf_keeps_valid_unlimited_objective_projection():
    mop = DPF1(M=10, D=2, K=5)
    assert mop.N == 6
    assert mop.evaluation(np.full((2, mop.N), 0.5))["F"].shape == (2, 10)


def test_other_dtlz_front_sampler_remains_available_for_many_objectives():
    front = DTLZ2(M=5).pf(n_points=32)
    assert front.shape == (32, 5)
