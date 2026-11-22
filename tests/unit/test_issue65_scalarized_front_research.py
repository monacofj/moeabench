# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent regression checks for the experimental global g scalarizer."""

from pathlib import Path
import runpy

import numpy as np
import pytest

from moeabench.mops.DTLZ5 import DTLZ5
from moeabench.mops.DTLZ6 import DTLZ6


_functions = runpy.run_path(
    str(Path(__file__).resolve().parents[2] /
        "scripts/audit/dtlz56_scalarized_front.py")
)
scalarized_candidate = _functions["scalarized_candidate"]
decision_vector = _functions["decision_vector"]
relaxed_feasible = _functions["_relaxed_feasible"]


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_scalarizer_finds_reproducible_feasible_nonzero_g_candidate(cls):
    """The known g=0 curve is not enough for the original 4-objective problem."""
    mop = cls(M=4)
    weights = np.array([
        0.571031078359263, 0.2152582100639811,
        0.6941304869517975, 0.3817987396940551,
    ])
    g_max = (mop.K / 4.0) if cls is DTLZ5 else float(mop.K)
    a = scalarized_candidate(weights, g_max, abs_tol=2e-5)
    b = scalarized_candidate(weights, g_max, abs_tol=2e-5)

    assert 0.2 < a.g < 1.0
    assert a.abs_gap < 2.1e-5
    assert a.nodes < 12000
    np.testing.assert_array_equal(a.objectives, b.objectives)
    assert a.g == b.g
    assert a.scale == b.scale

    x = decision_vector(a, M=mop.M, K=mop.K, problem=cls.__name__)
    assert x.shape == (mop.N,)
    assert np.all((0.0 <= x) & (x <= 1.0))
    independent_evaluation = mop.evaluation(x[None, :])["F"][0]
    np.testing.assert_allclose(
        independent_evaluation, a.objectives, rtol=2e-9, atol=2e-9
    )

    # A continuous g=0 front cannot dominate this nonzero-g candidate.
    # This is distinct from proving global Pareto optimality.
    assert not relaxed_feasible(a.objectives, 1.0, 0.0)

    # The independent fixed-g oracle confirms candidate feasibility.
    assert relaxed_feasible(a.objectives + 1e-10, 1.0 + a.g, a.g)


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_interval_relaxation_contains_every_fixed_g_spherical_patch(cls):
    """The whole interval relaxation must include each original valid point."""
    mop = cls(M=5)
    rng = np.random.default_rng(2026)
    for g in (0.0, 0.02, 0.11, 0.25, 0.6):
        k = mop.K
        x = np.full((1, mop.N), 0.5)
        x[0, :mop.M-1] = rng.random(mop.M-1)
        x[0, mop.M-1:] = (
            0.5 + np.sqrt(g/k) if cls is DTLZ5 else (g/k)**10
        )
        original = mop.evaluation(x)["F"][0]
        left = max(0.0, g - 0.1)
        right = min(mop.K/4 if cls is DTLZ5 else mop.K, g + 0.1)
        assert relaxed_feasible(original + 1e-9, 1.0 + left, right)


def test_scalarizer_parameter_validation():
    with pytest.raises(ValueError, match="strictly positive"):
        scalarized_candidate([1, 0, 1, 1], 2.5)
    with pytest.raises(ValueError, match="positive and finite"):
        scalarized_candidate([1, 1, 1, 1], -0.1)
    with pytest.raises(RuntimeError, match="not certified"):
        scalarized_candidate(
            [0.571031078359263, 0.2152582100639811,
             0.6941304869517975, 0.3817987396940551],
            2.5, abs_tol=1e-12, max_nodes=1
        )
