# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Paired research-only original DTLZ5/6 proof-gated source selection comparison.

Reuses independently established original M4/K1 448-candidate proof reach
and the original source maximin objective normalization. All comparisons
are OBSERVATIONAL and finite-source only. Never replace production
mop.ps(), alter accepted ADR, baseline or regression tests.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from .dtlz5_final_selection_trace import trace_original_dtlz5_selection
from .dtlz5_pool_proof_reach import (
    evaluate_proof_reach,compare_proof_gated_maximin,
)
from .dtlz6_pool_proof_reach import audit_source as audit_dtlz6_source

SCHEMA="DTLZ56-M4K1-PAIRED-PROOF-GATED-SELECTION-REVIEW-V1"


def assess_both_proof_gated_selections():
    source5=trace_original_dtlz5_selection(include_candidate_pool=True)
    reach5=evaluate_proof_reach(source5)
    ordinary5=compare_proof_gated_maximin(source5,reach5)
    stratified5=compare_proof_gated_maximin(
        source5,reach5,require_low_g=True)
    report6=audit_dtlz6_source(n_points=16)
    alternative6=report6["counterfactual"]

    if (reach5["source_candidate_count"]!=448 or
        report6["n_source_candidates"]!=448 or
        reach5["source_selection_count"]!=16 or
        report6["n_selected"]!=16):
        raise AssertionError("Original source/budget changed")
    if (ordinary5["status"]!="observational_proved_only_maximin" or
        stratified5["status"]!="observational_proved_only_stratified_maximin"
        or alternative6["status"]!="observational_proved_only_counterfactual"):
        raise ArithmeticError("Insufficient independently proved source candidates")
    if not (ordinary5["first_four_anchors_unchanged"] and
            stratified5["first_four_anchors_unchanged"]):
        raise AssertionError("Original DTLZ5 gzero anchor prefix changed")
    if not all(alt=="proved_pf" for alt in
               ordinary5["selected_proof_statuses"]+
               stratified5["selected_proof_statuses"]):
        raise AssertionError("Unproved DTLZ5 alternative selected source")
    if not (alternative6["all_nonanchor_exact_pf_members"] and
            alternative6["selected_count"]==16):
        raise AssertionError("Unproved DTLZ6 alternative selected source")
    if not (ordinary5["proof_gated_selection"]["positive_g_count"]>0 and
            stratified5["proof_gated_selection"]["positive_g_count"]>0 and
            alternative6["positive_g_selected_count"]>0):
        raise AssertionError("Alternative lacks known positive-g PF subset")
    original5=reach5["selected_counts"]
    original6=report6["selected_counts"]
    if (original5["dominated"]<1 or original6["unverified"]<2):
        raise ArithmeticError("Expected to observe historical scientific defects")

    def source_cover_summary(result):
        old=result["original_proved_source_pool_cover_radius"]
        new=result["counterfactual_proved_source_pool_cover_radius"]
        if not all(np.isfinite(v) and v>=0 for v in (old,new)):
            raise ArithmeticError("Invalid source-only observational radius")
        return {
            "original_radius":old,
            "alternative_radius":new,
            "alternative_over_original_ratio":(
                new/old if old>0 else None
            ),
            "distance_scope":"only_same_448_source_proved_PF_candidates",
            "continuous_original_PF_coverage":"unverified",
        }

    return {
        "schema":SCHEMA,
        "original_configurations":[
            {"problem":"DTLZ5","M":4,"K":1,
             "original_candidate_source":448,"original_selected_budget":16},
            {"problem":"DTLZ6","M":4,"K":1,
             "original_candidate_source":448,"original_selected_budget":16},
        ],
        "DTLZ5":{
            "original_pool_counts":reach5["pool_counts"],
            "original_selected_counts":original5,
            "known_independently_dominated_selected_indices":[10],
            "alternative_proved_source_candidate_indices":
                ordinary5["proved_source_candidate_indices"],
            "alternative_stratified_source_candidate_indices":
                stratified5["proved_source_candidate_indices"],
            "ordinary_observation":{
                "original_selection":ordinary5["original_selection"],
                "alternative_selection":ordinary5["proof_gated_selection"],
            },
            "low_g_stratified_observation":{
                "original_selection":stratified5["original_selection"],
                "alternative_selection":stratified5["proof_gated_selection"],
            },
            "alternative_all_selected_PF_membership":
                "individually_proved_under_declared_sufficient_oracle",
            "complete_original_PF_coverage":"unverified",
        },
        "DTLZ6":{
            "original_pool_counts":report6["pool_counts"],
            "historical_original_selected_counts":original6,
            "new_independently_dominated_selected_indices":[7,11],
            "three_still_unverified_selected_indices":[6,10,12],
            "alternative_proved_source_candidate_indices":
                alternative6["selected_proved_candidate_indices"],
            "alternative_selected_positive_g":
                alternative6["positive_g_selected_count"],
            "finite_proved_source_cover":source_cover_summary(alternative6),
            "alternative_all_selected_PF_membership":
                "individually_proved_under_declared_sufficient_oracle",
            "complete_original_PF_coverage":"unverified",
        },
        "production_sampler_modified":False,
        "accepted_ADR_modified":False,
        "existing_regression_tests_modified":False,
        "original_IGD_IGDplus_HV_fitness":"unverified",
        "new_IGD_IGDplus_HV_fitness":"unverified",
        "environment_result_certification":"unverified",
        "scientific_certification":"unverified",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",type=Path,required=True)
    args=p.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("No overwrite of research evidence")
    r=assess_both_proof_gated_selections()
    with args.report.open("x",encoding="utf-8") as stream:
        json.dump(r,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    for problem in ("DTLZ5","DTLZ6"):
        print(problem,"historical original pool:",r[problem]["original_pool_counts"])
        print(problem,"proof-gated substitute indices:",
              r[problem]["alternative_proved_source_candidate_indices"])
        print(problem,"scientific certificate",r["scientific_certification"])
    print("DTLZ6 source-only coverage comparison",
          r["DTLZ6"]["finite_proved_source_cover"])


if __name__=="__main__":
    main()
