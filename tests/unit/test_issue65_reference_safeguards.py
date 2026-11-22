# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused safeguards for known high-dimensional analytical-reference limitations."""

import numpy as np
import pytest

from moeabench.mops import DTLZ2, DTLZ5, DTLZ6, DPF1, DPF5


@pytest.mark.parametrize("mop_cls", [DTLZ5, DTLZ6])
@pytest.mark.parametrize("M", [4, 5, 10])
def test_dtlz56_canonical_full_reference_includes_nonzero_g(mop_cls, M):
    """Cover original nondegenerate g>0 regions and the historical g=0 curve."""
    mop = mop_cls(M=M)
    ps = mop.ps(n_points=32)
    ps_again = mop.ps(n_points=32)
    np.testing.assert_array_equal(ps, ps_again)
    front = mop.pf(n_points=32)
    np.testing.assert_allclose(front, mop.evaluation(ps)["F"], atol=1e-10, rtol=0)
    assert ps.shape == (32, mop.N)
    assert front.shape == (32, M)
    assert np.isfinite(front).all()
    assert np.unique(np.round(front, 10), axis=0).shape[0] == 32
    distance = ps[:, M-1:]
    g = (
        np.sum((distance - 0.5)**2, axis=1)
        if mop_cls is DTLZ5
        else np.sum(distance**0.1, axis=1)
    )
    assert np.any(g == 0)
    assert np.count_nonzero(g > 1e-6) >= 1
    np.testing.assert_allclose(
        np.linalg.norm(front, axis=1), 1.0 + g, atol=2e-10, rtol=0
    )


@pytest.mark.parametrize("mop_cls", [DTLZ5, DTLZ6])
@pytest.mark.parametrize("M", [2, 3])
def test_dtlz56_valid_legacy_front_has_no_incompleteness_warning(mop_cls, M):
    mop = mop_cls(M=M)
    front = mop.pf(n_points=32)
    assert front.shape == (32, M)
    np.testing.assert_allclose(np.sum(front**2, axis=1), 1.0, atol=1e-12, rtol=0)


@pytest.mark.parametrize("M,N", [(7, 6), (10, 6), (12, 8)])
def test_dpf5_rejects_explicit_insufficient_dimensions_at_construction(M, N):
    with pytest.raises(ValueError, match=r"DPF5 requires N >= M"):
        DPF5(M=M, N=N, D=2, K=5)


@pytest.mark.parametrize("M,D,K", [(3, 2, 5), (6, 2, 5), (7, 2, 5), (10, 2, 5), (10, 3, 5)])
def test_dpf5_default_dimensions_include_x_m(M, D, K):
    mop = DPF5(M=M, D=D, K=K)
    expected_n = max(D + K - 1, M)
    assert mop.N == expected_n
    ps = mop.ps(n_points=16)
    front = mop.evaluation(ps)["F"]
    assert ps.shape == (16, expected_n)
    assert front.shape == (16, M)
    assert np.isfinite(front).all()
    np.testing.assert_array_equal(ps[:, M - 1], ps[:, 0])
    if expected_n > M:
        np.testing.assert_array_equal(ps[:, M:], np.full((16, expected_n - M), 0.5))


@pytest.mark.parametrize("M", [7, 10])
def test_dpf5_works_with_explicit_sufficient_decision_variables(M):
    mop = DPF5(M=M, N=M + 2, D=2, K=5)
    ps = mop.ps(n_points=16)
    front = mop.evaluation(ps)["F"]
    assert ps.shape == (16, M + 2)
    assert front.shape == (16, M)
    assert np.isfinite(front).all()


def test_other_dpf_keeps_valid_unlimited_objective_projection():
    mop = DPF1(M=10, D=2, K=5)
    assert mop.N == 6
    assert mop.evaluation(np.full((2, mop.N), 0.5))["F"].shape == (2, 10)


def test_other_dtlz_front_sampler_remains_available_for_many_objectives():
    front = DTLZ2(M=5).pf(n_points=32)
    assert front.shape == (32, 5)
