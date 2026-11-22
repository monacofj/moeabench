# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent checks for corrected original DTLZ5/6 full-front references."""

import numpy as np
import pytest

from moeabench.mops import DTLZ5, DTLZ6
from moeabench.mops._dtlz56_front import (
    _bounded_scale,
    _relaxed_feasible,
    decision_vector,
    scalarized_candidate,
)


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
@pytest.mark.parametrize("M", [4, 5, 10])
def test_complete_gt_is_feasible_unique_and_contains_nonzero_g(cls, M):
    mop = cls(M=M)
    x = mop.ps(n_points=64)
    front = mop.pf(n_points=64)
    np.testing.assert_array_equal(mop.ps(n_points=64), x)
    np.testing.assert_allclose(front, mop.evaluation(x)["F"], atol=2e-10, rtol=0)
    assert x.shape == (64, mop.N)
    assert front.shape == (64, M)
    assert np.isfinite(front).all()
    assert np.unique(np.round(front, 9), axis=0).shape[0] == 64

    tail = x[:, M-1:]
    g = np.sum((tail-0.5)**2, axis=1) if cls is DTLZ5 else np.sum(tail**0.1, axis=1)
    assert np.count_nonzero(g == 0.0) >= 2
    assert np.count_nonzero(g > 1e-5) >= 1
    np.testing.assert_allclose(np.linalg.norm(front, axis=1), 1.0+g, atol=2e-10)

    # Independent, full pairwise dominance (finite-sample necessary invariant).
    for i in range(len(front)):
        dominates = (
            np.all(front <= front[i] + 2e-9, axis=1)
            & np.any(front < front[i] - 2e-9, axis=1)
        )
        assert not np.any(dominates), f"Sample {i} is dominated"


def test_closed_form_angular_capacity_matches_independent_box_feasibility():
    """Compare closed-form formula against independent original greedy oracle."""
    rng = np.random.default_rng(1517)
    for M in (4, 5, 10):
        for _ in range(35):
            weights = rng.uniform(0.03, 1.0, M)
            g_low = float(rng.uniform(0, 0.8))
            g_high = float(rng.uniform(g_low, 1.7))
            lb, ub = _bounded_scale(weights, g_low, g_high)
            assert 0 <= lb <= ub
            assert ub-lb < 3e-12
            assert _relaxed_feasible(
                (ub + 5e-8)*weights, 1.0+g_low, g_high
            )
            assert not _relaxed_feasible(
                (lb - 5e-8)*weights, 1.0+g_low, g_high
            )


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_scalarization_global_bound_reconstructs_original_problem(cls):
    mop = cls(M=4)
    w = np.array([0.571031078359263, 0.2152582100639811,
                  0.6941304869517975, 0.3817987396940551])
    g_max = mop.K / 4 if cls is DTLZ5 else mop.K
    result = scalarized_candidate(w, g_max, abs_tol=1e-8)
    assert 0.2 < result.g < 1.0
    assert result.abs_gap < 1.01e-8
    assert result.scale - result.lower_bound <= 1.01e-8
    x = decision_vector(result, M=mop.M, K=mop.K, problem=cls.__name__)
    np.testing.assert_allclose(
        mop.evaluation(x[None, :])["F"][0],
        result.objectives, atol=3e-10, rtol=0
    )
    # The continuous g=0 curve cannot dominate this objective vector.
    assert not _relaxed_feasible(result.objectives, 1.0, 0.0)


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_original_three_objective_curve_is_unchanged(cls):
    mop = cls(M=3)
    ps = mop.ps(n_points=64)
    front = mop.evaluation(ps)["F"]
    assert np.allclose(np.linalg.norm(front, axis=1), 1.0)


def test_ill_conditioned_positive_weights_reconstruct_feasible_original_vector():
    """Regression for a sparse high-dimensional preference direction."""
    w = np.array([
        1.58310251e-4, 6.24477665e-9, 7.45064597e-5,
        1.30502082e-5, 1.21715514e-1, 5.18506217e-4,
        1.80935861e-9, 2.16811697e-6, 5.04378037e-4,
        6.27287250e-14,
    ])
    mop = DTLZ6(M=10)
    # The scalarization contract rejects silently modified preferences;
    # the finite GT procedure conditions these rays explicitly.
    with pytest.raises(ValueError, match="Ill-conditioned"):
        scalarized_candidate(w, max_g=mop.K, abs_tol=1e-7)
    conditioned = np.maximum(w, 1e-7 * np.max(w))
    candidate = scalarized_candidate(conditioned, max_g=mop.K, abs_tol=1e-7)
    assert np.isfinite(candidate.objectives).all()
    assert candidate.abs_gap < 1.01e-7
    x = decision_vector(candidate, M=mop.M, K=mop.K, problem="DTLZ6")
    np.testing.assert_allclose(
        mop.evaluation(x[None, :])["F"][0],
        candidate.objectives, atol=3e-10, rtol=0,
    )
