# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive tests for observational low-positive-g stratum preservation."""
from scripts.audit.dtlz5_pool_proof_reach import (
    evaluate_proof_reach, compare_proof_gated_maximin,
)
from scripts.audit.dtlz5_row10_exact_dominance import SOURCE_ROW_10


def _source_with_proved_low_g():
    anchors=[
        (0.0,.5,.5,.5),(1/3,.5,.5,.5),
        (2/3,.5,.5,.5),(1.0,.5,.5,.5),
    ]
    # x2=x3=1 is an exact support stratum. A near .5 tail
    # supplies a strictly positive g <= 1/16.
    low=(0.20,1.0,1.0,.60)
    high=(0.20,1.0,1.0,1.0)
    rows=[SOURCE_ROW_10]*100+[low]*100+[high]*248
    source={
        "candidate_pool_decision_hex":[
            [float(v).hex() for v in r] for r in rows
        ],
        "candidate_pool_source_families":(
            ["sobol"]*256+["gaussian"]*128+["gamma"]*64),
        "selected_rows":[{
            "selected_index":i,
            "decision_hex":[float(v).hex() for v in r],
            "trace":None,
        } for i,r in enumerate(anchors+[SOURCE_ROW_10]*12)],
    }
    return source


def test_explicit_small_positive_g_persists_with_a_valid_proof():
    source=_source_with_proved_low_g()
    reach=evaluate_proof_reach(source)
    result=compare_proof_gated_maximin(
        source,reach,require_low_g=True)
    assert result["status"]=="observational_proved_only_stratified_maximin"
    assert result["first_four_anchors_unchanged"] is True
    assert result["selected_proof_statuses"] == ["proved_pf"]*12
    assert result["experimental_low_g_stratum_required"] is True
    assert result["proof_gated_selection"]["positive_g_min"] <= 1/16
    assert result["scientific_certification"]=="unverified"


def test_no_proved_low_positive_g_fails_closed():
    source=_source_with_proved_low_g()
    source["candidate_pool_decision_hex"][100:200]=[
        source["candidate_pool_decision_hex"][200]
    ]*100
    reach=evaluate_proof_reach(source)
    result=compare_proof_gated_maximin(
        source,reach,require_low_g=True)
    assert result["status"]=="unverified_missing_low_g_proved_component"
    assert result["scientific_certification"]=="unverified"
