# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive original proof-gated alternative comparison contract tests."""
from scripts.audit.dtlz56_paired_proof_gated_replacement import (
    assess_both_proof_gated_selections,
)


def test_original_paired_gated_selection_preserves_science_limits():
    r=assess_both_proof_gated_selections()
    assert r["schema"]=="DTLZ56-M4K1-PAIRED-PROOF-GATED-SELECTION-REVIEW-V1"
    assert [x["problem"] for x in r["original_configurations"]]==[
        "DTLZ5","DTLZ6"
    ]
    assert all(x["original_candidate_source"]==448 and
               x["original_selected_budget"]==16
               for x in r["original_configurations"])
    assert r["DTLZ5"]["known_independently_dominated_selected_indices"]==[10]
    assert r["DTLZ6"]["new_independently_dominated_selected_indices"]==[7,11]
    assert r["DTLZ6"]["three_still_unverified_selected_indices"]==[6,10,12]
    for problem in ("DTLZ5","DTLZ6"):
        assert len(r[problem]["alternative_proved_source_candidate_indices"])==12
        assert len(set(r[problem]["alternative_proved_source_candidate_indices"]))==12
        assert r[problem]["complete_original_PF_coverage"]=="unverified"
    assert r["DTLZ5"]["low_g_stratified_observation"][
        "alternative_selection"]["positive_g_count"]>0
    assert r["DTLZ6"]["alternative_selected_positive_g"]>0
    assert r["DTLZ6"]["finite_proved_source_cover"][
        "continuous_original_PF_coverage"]=="unverified"
    assert r["original_IGD_IGDplus_HV_fitness"]=="unverified"
    assert r["new_IGD_IGDplus_HV_fitness"]=="unverified"
    assert r["scientific_certification"]=="unverified"
    assert not r["production_sampler_modified"]
    assert not r["accepted_ADR_modified"]
    assert not r["existing_regression_tests_modified"]
