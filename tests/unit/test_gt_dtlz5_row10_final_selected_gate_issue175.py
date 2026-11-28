# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive final-selected scientific gate for exact archived off-ray witness."""
import numpy as np

from scripts.audit.dtlz56_final_selected_small_gt import analyze_final_selected
from scripts.audit.dtlz56_paired_gt_evidence import _original_objectives
from scripts.audit.dtlz5_row10_exact_dominance import SOURCE_ROW_10


def test_original_dtlz5_repaired_archived_row_is_scientifically_blocked():
    x = np.array([
        (0.0, 1.0, 0.0, 0.5),
        SOURCE_ROW_10,
    ])
    f, _ = _original_objectives(x, problem="DTLZ5", m=4, k=1)
    out = analyze_final_selected(x, f, problem="DTLZ5", m=4, k=1)
    assert out["exact_rational_g_decision_image_ledger"]["rows"][1][
        "status"] == "unverified"
    assert out["proven_dominated_decision_row_indices"] == [1]
    assert out["experimental_reference_gate"] == (
        "blocked_proven_dominated_decision")
    row = out["exact_off_ray_dominance_witnesses"][0]
    assert row["row"] == 1
    assert row["certificate"]["conclusion"] == (
        "rigorously_strictly_dominated")
    assert out["scientific_certification"] == "unverified"


def test_off_ray_witness_is_never_assumed_for_nearby_different_sources():
    x = np.array([
        (0.0, 1.0, 0.0, 0.5),
        SOURCE_ROW_10,
    ])
    # One ULP different: the archived source-bound proof must not be reused.
    x[1, 0] = np.nextafter(x[1, 0], 1.0)
    f, _ = _original_objectives(x, problem="DTLZ5", m=4, k=1)
    out = analyze_final_selected(x, f, problem="DTLZ5", m=4, k=1)
    assert out["exact_off_ray_dominance_witnesses"] == []
    assert out["scientific_certification"] == "unverified"
