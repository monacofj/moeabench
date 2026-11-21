# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent DTLZ8 published-front membership diagnostics."""
import numpy as np
import pytest

from scripts.audit.gt_dtlz8_candidate_issue119 import candidate_objectives
from scripts.audit.gt_dtlz8_membership_issue119 import membership_residual


@pytest.mark.parametrize("m", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("strategy", ["halton", "lattice"])
def test_dtlz8_sampled_candidates_are_on_published_pareto_geometry(m, strategy):
    f = candidate_objectives(m, 512, patch_strategy=strategy)
    assert membership_residual(f).max() < 1e-13


def test_dtlz8_membership_rejects_feasible_but_dominated_objectives():
    # On M=3 with t=0.5, the Pareto-line coordinate is (1-t)/4=0.125.
    # Increasing f1 while keeping other objectives fixed preserves
    # feasibility but yields a dominated off-front objective vector.
    f = np.array([[0.20, 0.125, 0.5]])
    assert membership_residual(f)[0] > 0.07


def test_dtlz8_membership_rejects_unsupported_dimension():
    with pytest.raises(ValueError, match="M>=2"):
        membership_residual(np.zeros((4, 1)))
