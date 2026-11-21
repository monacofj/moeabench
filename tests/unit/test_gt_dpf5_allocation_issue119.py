# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New read-only fixed-total P/D allocation research assertions."""
import pytest
from scripts.audit.gt_dpf5_allocation_issue119 import summarize


@pytest.mark.parametrize("m,d",[(3,2),(5,2),(6,3),(6,4)])
def test_all_allocations_keep_fixed_branchwise_unique_budget(m,d):
    result=summarize(m,d,128,sampler_seeds=(71,72),probe_seeds=(201,202),probe_count=64)
    assert not result["normative_mixed_dimensional_measure_adopted"]
    assert not result["calibration_default_authorized"]
    assert result["finite_probe_max_not_certified_worst_case"]
    n=result["fixed_branchwise_total"]
    for item in result["records"]:
        assert item["P_count"]+item["D_count"]==n
        assert item["branchwise_total"]==n
        assert item["global_exact_unique"]<=n
        assert item["max_conditional_probe_distance"]>=0


def test_allocations_use_corrected_effective_branch_counts():
    r=summarize(3,2,128,sampler_seeds=(71,),probe_seeds=(201,),probe_count=64)
    assert r["incumbent_returned_decisions"] == 128
    assert r["allocations"]["incumbent"]["P"] > 0
    assert r["allocations"]["incumbent"]["D"] > 0
    assert sum(r["allocations"]["incumbent"].values()) == 128
    assert r["fixed_branchwise_total"] == 128
    assert len(r["records"]) == 3
