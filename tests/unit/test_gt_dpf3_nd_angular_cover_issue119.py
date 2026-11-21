# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New separate angular-grid geometric certificate checks (issue 119)."""
import pytest
from scripts.audit.gt_dpf3_nd_angular_cover_issue119 import audit


@pytest.mark.parametrize("m,d,k",[(3,2,17),(5,2,33),(5,3,4),(5,3,7),(6,4,3),(6,4,5)])
def test_published_nd_tensor_grid_bound_and_rounding_distinction(m,d,k):
    pytest.importorskip("mpmath")
    r=audit(m,d,k,128)
    assert r["requested_tensor_points"]==k**(d-1)
    assert r["float_exact_unique_objectives"]<=r["requested_tensor_points"]
    assert r["observed_probe_max"]<=r["analytic_real_Hausdorff_upper_bound"]+2e-12
    assert r["high_precision_angle_roundtrip_max_norm_estimate"]<1e-10
    assert r["float_published_objective_roundtrip_max_norm"]<1e-10
    assert r["theorem_for_published_continuous_PF"]
    assert not r["rigorous_float_rounding_enclosure"]
    assert not r["calibration_default_approved"]


def test_dpf3_angular_certification_improves_as_grid_refines():
    pytest.importorskip("mpmath")
    coarse=audit(5,3,4,128)
    fine=audit(5,3,8,128)
    assert fine["analytic_real_Hausdorff_upper_bound"] < coarse["analytic_real_Hausdorff_upper_bound"]
