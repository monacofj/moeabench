# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent tests for exact fixed-g DTLZ5/6 feasibility diagnostic."""

from pathlib import Path
import runpy

import numpy as np
import pytest

from moeabench.mops.DTLZ5 import DTLZ5
from moeabench.mops.DTLZ6 import DTLZ6


_ORACLE = runpy.run_path(
    str(Path(__file__).resolve().parents[2] /
        "scripts/audit/dtlz56_fixed_g_dominance_oracle.py")
)["can_weakly_dominate_at_g"]


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_exact_fixed_g_oracle_accepts_feasible_original_evaluation(cls):
    """Every valid objective vector must be feasible at its original g."""
    mop = cls(M=4)
    k = mop.N - mop.M + 1
    g = 0.328135
    x = np.full((1, mop.N), 0.5)
    x[0, :3] = [0.173406, 0.976618, 0.066887]
    x[0, 3:] = (
        0.5 + np.sqrt(g/k) if cls is DTLZ5 else (g/k)**10
    )
    y = mop.evaluation(x)["F"][0]
    assert _ORACLE(y, g)
    assert not _ORACLE(y, 0.0)  # Even the entire continuous g=0 curve fails.


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_fixed_g_oracle_rejects_impossible_lower_objective_vector(cls):
    mop = cls(M=4)
    x = np.full((1, mop.N), 0.5)
    x[0, 3:] = 0.5 if cls is DTLZ5 else 0.0
    original = mop.evaluation(x)["F"][0]
    assert not _ORACLE(0.5 * original, 0.0)


def test_fixed_g_oracle_validates_domain():
    with pytest.raises(ValueError, match="nonnegative"):
        _ORACLE([0.5, 0.5, 0.5, 0.5], -0.1)
