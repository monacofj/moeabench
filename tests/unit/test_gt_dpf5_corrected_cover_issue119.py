# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent source-oracle and conditional tensor-grid coverage for DPF5."""
import pytest
from scripts.audit.gt_dpf5_corrected_cover_issue119 import audit


@pytest.mark.parametrize("m,d",[(3,2),(5,2),(6,3),(6,4)])
def test_corrected_dpf5_has_independent_published_objectives_and_angular_coverage(m,d):
    row=audit(m,d,128,probe_count=128)
    assert row["returned"] == row["global_numerical_unique"] == 128
    assert row["max_production_vs_independent_published"]<5e-13
    assert row["conditional_exact_real_cover_bound"] is not None
    assert not row["automatic_calibration_budget_chosen"]
    for part in row["branches"].values():
        assert part["largest_completed_tensor_grid_k"]>=2
        assert part["independent_probe_max"]<=part["ideal_real_angular_cover_bound"]+1e-12
        assert not part["full_continuous_float64_coverage_certified"]
