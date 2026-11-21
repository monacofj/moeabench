# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New read-only branchwise budget comparison tests, issue 119."""
import pytest
from scripts.audit.gt_dpf5_branch_sampler_compare_issue119 import compare


@pytest.mark.parametrize("m,d",[(3,2),(5,2),(6,3),(6,4)])
def test_branchwise_effective_budget_frozen_without_mixture_choice(m,d):
    r=compare(m,d,128,seed=71,probe_seeds=(201,202),probes_per_branch=64)
    assert r["candidate_branch_allocation_is_incumbent"]
    assert not r["release_GT_approved"]
    assert not r["certified_supremum_or_uniformity"]
    for b in ("P","D"):
        record=r["branches"][b]
        assert record["effective_unique_budget"]>0
        assert len(record["probe_results"])==2
        assert record["conditional_measure_not_selected"]
        for probe in record["probe_results"]:
            for v in probe["measurements"].values():
                assert 0<=v["median"]<=v["p95"]<=v["max_observed"]


def test_comparison_reproducible_distances_independent_of_timing():
    a=compare(5,2,128,seed=71,probe_seeds=(201,),probes_per_branch=64)
    b=compare(5,2,128,seed=71,probe_seeds=(201,),probes_per_branch=64)
    for name in ("P","D"):
        assert a["branches"][name]["probe_results"]==b["branches"][name]["probe_results"]
