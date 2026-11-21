# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New research-only DPF3 published projection and global PS witness tests."""
import pytest
from scripts.audit.gt_dpf3_global_membership_issue119 import audit


@pytest.mark.parametrize("m,d",[(3,2),(5,2),(6,3),(7,4)])
def test_published_dpf3_sorted_projection_order_embedding_and_witness(m,d):
    r=audit(m,d,64)
    assert r["max_projection_telescope_error"]<1e-12
    assert r["max_essential_unit_sphere_error"]<1e-12
    assert r["all_positive_g_witness_componentwise"]
    assert r["all_positive_g_witness_strict"]
    assert not r["current_implementation_certified"]
    assert not r["finite_reference_sampling_certified"]
