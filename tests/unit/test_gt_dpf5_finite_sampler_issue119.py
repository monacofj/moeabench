# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New finite DPF5 sampler diagnostics only. No existing regressions edited."""
import pytest
from scripts.audit.gt_dpf5_finite_sampler_issue119 import stats


@pytest.mark.parametrize("m,d",[(3,2),(5,2),(6,3),(6,4)])
def test_conditional_coverages_are_reproducible_and_not_claimed_certified(m,d):
    a=stats(m,d,64,64,3101)
    b=stats(m,d,64,64,3101)
    assert a==b
    assert a["branch_counts"]["P"]["decision_count"]+a["branch_counts"]["D"]["decision_count"]==a["returned_ps"]
    assert a["published_objective_unique_exact"]<=a["returned_ps"]
    assert not a["finite_support_coverage_certified"]
    assert not a["sampling_measure_selected"]


def test_m3_actual_grid_cardinality_and_degenerate_objective_multiplicity():
    a=stats(3,2,128,64)
    assert a["returned_ps"]==121
    assert a["branch_counts"]["D"]["published_objective_duplicates_exact"]>0
    assert a["published_objective_unique_exact"]<a["returned_ps"]


def test_branch_specific_probe_distances_are_nonnegative_and_finite():
    row=stats(5,2,64,64)
    for key in ("P","D"):
        part=row["conditional_branch_coverage"][key]
        assert 0<=part["probe_to_sample_median"]<=part["probe_to_sample_p95"]<=part["probe_to_sample_max"]
