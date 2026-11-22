# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent continuous g=0 domination oracle for original DTLZ5/6 (#65).

This proves incompleteness of the g=0 curve as a candidate dominator.
It does not claim that a g>0 witness is globally Pareto optimal.
"""

import numpy as np
import pytest

from moeabench.mops.DTLZ5 import DTLZ5
from moeabench.mops.DTLZ6 import DTLZ6


def _g_zero_curve_can_weakly_dominate(f):
    """Exact continuous-curve test; no sampled reference grid involved.

    The degenerate g=0 curve has components a[j]*cos(t) for j<M-1
    and sin(t) for j=M-1, 0<=t<=pi/2. Of all t with sin(t)<=f[-1],
    the largest t minimizes all preceding coordinates.
    """
    m = len(f)
    scale = np.array([2.0 ** (-(m-2)/2)] +
                     [2.0 ** (-(m-j-1)/2) for j in range(1, m-1)])
    best_cos = np.sqrt(max(0.0, 1.0 - min(1.0, f[-1])**2))
    return f[-1] >= 0 and np.all(f[:-1] >= scale * best_cos - 1e-12)


@pytest.mark.parametrize("problem_cls", [DTLZ5, DTLZ6])
def test_g_positive_witness_is_not_dominated_by_any_point_on_g_zero_curve(problem_cls):
    """Use the production evaluator, but an independent analytic dominator."""
    mop = problem_cls(M=4)
    x = np.full((1, mop.N), 0.5)
    x[0, :3] = [0.173406, 0.976618, 0.066887]
    g = 0.328135
    k = mop.N - mop.M + 1
    x[0, 3:] = (0.5 + np.sqrt(g / k)
                if problem_cls is DTLZ5 else (g / k)**10)
    observed = mop.evaluation(x)["F"][0]
    assert np.isfinite(observed).all()
    assert not _g_zero_curve_can_weakly_dominate(observed)


@pytest.mark.parametrize("problem_cls", [DTLZ5, DTLZ6])
def test_original_evaluator_stays_available_when_incomplete_gt_is_blocked(problem_cls):
    mop = problem_cls(M=4)
    with pytest.raises(NotImplementedError, match="full Ground Truth"):
        mop.pf(n_points=32)
    evaluated = mop.evaluation(np.full((2, mop.N), 0.5))["F"]
    assert evaluated.shape == (2, mop.M)
    assert np.isfinite(evaluated).all()
