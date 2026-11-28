# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive research tests for proved-only same-budget maximin comparison."""
import numpy as np

from scripts.audit.dtlz5_pool_proof_reach import (
    evaluate_proof_reach, compare_proof_gated_maximin,
)
from scripts.audit.dtlz5_row10_exact_dominance import SOURCE_ROW_10


def test_proof_only_maximin_retains_anchors_and_rejects_known_dominated():
    anchor=np.array([
        (0.0,0.5,0.5,0.5),
        (1/3,0.5,0.5,0.5),
        (2/3,0.5,0.5,0.5),
        (1.0,0.5,0.5,0.5),
    ])
    good=[
        (0.10+i/200,1.0,1.0,1.0)
        for i in range(50)
    ]
    source_rows=([SOURCE_ROW_10]*100+
                 good*6+
                 [(0.20,0.5,0.5,0.5)]*48)
    assert len(source_rows)==448
    selected=[*anchor]+[SOURCE_ROW_10]*12
    source={
        "candidate_pool_decision_hex":[
            [float(x).hex() for x in row] for row in source_rows
        ],
        "candidate_pool_source_families":[
            "sobol"]*256+["gaussian"]*128+["gamma"]*64,
        "selected_rows":[{
            "selected_index":i,
            "decision_hex":[float(x).hex() for x in row],
            "trace":None if i<4 else {"scalarizer_candidate_index":0},
        } for i,row in enumerate(selected)],
    }
    reach=evaluate_proof_reach(source)
    result=compare_proof_gated_maximin(source,reach)
    assert result["status"]=="observational_proved_only_maximin"
    assert result["first_four_anchors_unchanged"] is True
    assert len(result["proved_source_candidate_indices"])==12
    assert all(i>=100 for i in result["proved_source_candidate_indices"])
    assert result["selected_proof_statuses"]==["proved_pf"]*12
    assert result["proof_gated_selection"]["positive_g_count"] > 0
    assert result["scientific_certification"]=="unverified"
    assert result["continuous_front_coverage"]=="unverified"
    assert result["indicator_fitness"]=="unverified"


def test_insufficient_proof_candidates_abstains_without_selection():
    anchor=[(float(i)/3,.5,.5,.5) for i in range(4)]
    source_rows=[SOURCE_ROW_10]*448
    source={
        "candidate_pool_decision_hex":[
            [float(x).hex() for x in row] for row in source_rows
        ],
        "candidate_pool_source_families":[
            "sobol"]*256+["gaussian"]*128+["gamma"]*64,
        "selected_rows":[{
            "selected_index":i,
            "decision_hex":[float(x).hex() for x in row],
            "trace":None,
        } for i,row in enumerate(anchor+[SOURCE_ROW_10]*12)],
    }
    reach=evaluate_proof_reach(source)
    result=compare_proof_gated_maximin(source,reach)
    assert result["status"]=="unverified_insufficient_proved_candidates"
    assert result["scientific_certification"]=="unverified"
