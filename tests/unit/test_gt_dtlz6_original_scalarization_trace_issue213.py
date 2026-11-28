# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive original DTLZ6 source selected scalarizer provenance tests."""
from scripts.audit.dtlz6_original_selected_scalarization_trace import (
    trace_original_selected_dtlz6,
)


def test_original_dominated_selected_rows_reconstruct_exactly_from_source():
    proof=trace_original_selected_dtlz6()
    assert proof["source_candidate_count"]==448
    assert proof["selected_count"]==16
    assert proof["known_independently_dominated_selected_indices"]==[7,11]
    assert len(proof["specific_provenance_rows"])==2
    assert proof["causal_repair_of_sampler"]=="unverified"
    assert proof["whole_gt_membership"]=="rejected_by_independent_exact_proof"
    for row in proof["specific_provenance_rows"]:
        assert row["matched_source_candidate_indices"]
        assert row["scalarization_candidate_index"] is not None
        assert len(row["decision_hex"])==4
        assert len(row["angular_active_faces"])==2
