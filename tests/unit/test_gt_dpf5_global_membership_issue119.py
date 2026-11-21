# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New DPF5 global PF membership witness tests (no regression modifications)."""
import pytest
from scripts.audit.gt_dpf5_global_membership_issue119 import run_case


@pytest.mark.parametrize("m,d",[(3,2),(5,2),(6,3),(6,4),(8,3),(8,4)])
def test_each_feasible_positive_g_has_a_dominating_g_zero_preimage(m,d):
    r=run_case(m,d,64)
    assert r["positive_g_count"]>0
    assert r["witness_max_g"]==pytest.approx(0,abs=1e-16)
    assert r["max_unit_norm_error"]<1e-12
    assert r["max_radial_scaling_error"]<1e-12
    assert r["all_g_positive_weakly_dominated"]
    assert r["all_g_positive_strictly_dominated"]
    assert not r["finite_sampler_coverage_certified"]
