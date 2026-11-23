# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New research-only GT convergence checks; preserves existing regressions."""
import numpy as np
import pytest

from scripts.audit.hv_gt_refinement_issue89 import analyze_case


@pytest.mark.parametrize("name", ["DTLZ7", "DPF1"])
def test_nested_gt_monotonicity_under_common_geometry(name):
    study = analyze_case(
        name, dense_count=512, budgets=(16, 32, 64, 128),
        check_public=True,
    )
    rows = study["budgets"]
    assert study["nested_hv_nondecreasing"]
    assert all(np.isfinite(r["nested_hv"]) for r in rows)
    assert all(0 < r["nested_hv"] <= study["anchor_hv"] for r in rows)
    assert all(r["nested_gap_to_anchor"] >= -1e-12 for r in rows)
    assert all(r["direct_gap_to_anchor"] >= -1e-12 for r in rows)


def test_dtlz7_repeated_budget_is_the_same_stratified_prefix():
    study = analyze_case(
        "DTLZ7", dense_count=512, budgets=(16, 32, 64, 128)
    )
    for row in study["budgets"]:
        assert row["direct_hv"] == pytest.approx(
            row["nested_hv"], abs=1e-12
        )


def test_dpf1_recomputed_grid_is_not_assumed_to_be_nested():
    study = analyze_case(
        "DPF1", dense_count=512, budgets=(16, 32, 64, 128)
    )
    # The real sampler uses linspace(0,1,n), which repositions interior
    # sample points whenever n changes; its successive fronts are not
    # required to be nested. The nested strategy is explicitly separate.
    assert study["budgets"][0]["direct_hv"] != pytest.approx(
        study["budgets"][0]["nested_hv"], abs=1e-9
    )
