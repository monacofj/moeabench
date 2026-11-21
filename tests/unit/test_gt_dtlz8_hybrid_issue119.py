# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent positive checks for DTLZ8 hybrid line/surface research."""
import numpy as np
import pytest

from scripts.audit.gt_dtlz8_candidate_issue119 import candidate_objectives
from scripts.audit.gt_dtlz8_membership_issue119 import (
    feasibility_residual, membership_residual,
)
from scripts.audit.gt_dtlz8_policy_issue119 import summarize


@pytest.mark.parametrize("m", [2, 3, 5, 10])
@pytest.mark.parametrize("budget", [128, 512, 2000])
def test_hybrid_analytic_membership_budget_and_line_discretization(m, budget):
    hybrid = candidate_objectives(m, budget, patch_strategy="hybrid")
    halton = candidate_objectives(m, budget, patch_strategy="halton")
    assert hybrid.shape == (budget, m)
    assert len(np.unique(hybrid, axis=0)) == budget
    assert np.max(membership_residual(hybrid)) <= 1e-10
    assert np.max(feasibility_residual(hybrid)) <= 1e-10
    if m >= 3:
        # The hybrid explicitly retains the Halton line allocation.
        hybrid_line = hybrid[hybrid[:, -1] >= 1 / 3]
        halton_line = halton[halton[:, -1] >= 1 / 3]
        np.testing.assert_array_equal(hybrid_line, halton_line)
    else:
        np.testing.assert_array_equal(hybrid, halton)


@pytest.mark.parametrize("m", [2, 3, 5, 10])
def test_hybrid_policy_remains_research_only(m):
    evidence = summarize(m, 128, probes=32)
    assert evidence["admissibility"]["hybrid"]["count"] == 128
    assert evidence["admissibility"]["hybrid"]["unique"] == 128
    assert not evidence["production_policy_approved"]
    assert not evidence["canonical_mixed_dimensional_measure_defined"]
    assert set(evidence["per_component"]["line"]) == {
        "candidate", "lattice", "hybrid"
    }
