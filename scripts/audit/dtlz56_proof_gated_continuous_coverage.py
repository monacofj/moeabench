# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rigorous continuous-subset coverage comparison for proof-gated n16 alternatives.

For original M4 K1 DTLZ5/6, reconstruct BOTH unmodified original n16 PS
and the same 448 scalarized candidate decisions. Select an alternative
16-point PS using the already independent per-decision exact PF proof
classifiers and the unchanged source-normalized objective maximin policy.
Compute outward exact-rational original-equation one-sided cover upper
bounds from each of THREE declared, rigorously PF continuous subsets:
the complete g=0 Pareto curve, the x2=x3=1 full-radial support surface,
and the x2=1,x3=0 full-radial second support surface.

The result compares *rigorous bounds*, NOT exact Hausdorff distances.
A smaller upper bound does NOT by itself prove a smaller true covering
radius; the previous reference is scientifically invalid regardless.
No assertion that the union exhausts the complete original PF, no
indicator acceptance or production sampler change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from moeabench.mops import DTLZ6
from moeabench.mops import _dtlz56_front as front
from .dtlz5_final_selection_trace import trace_original_dtlz5_selection
from .dtlz5_pool_proof_reach import (
    evaluate_proof_reach,compare_proof_gated_maximin
)
from .dtlz6_pool_proof_reach import (
    audit_source as audit_dtlz6_source
)
from .dtlz56_paired_gt_evidence import _original_objectives
from .dtlz56_gzero_continuous_cover import certify_gzero_curve_cover
from .dtlz56_full_radial_face_actual_source import certify_full_radial_face_to_source
from .dtlz56_second_support_surface_actual_source import (
    certify_second_surface_to_original_ps
)
from .dtlz5_patch_actual_source_cover import _decode
from .dtlz56_exact_same_ray import _encode

SCHEMA="DTLZ56-ORIGINAL-M4K1-PROOF-GATED-CONTINUOUS-SUBSETS-REVIEW-V1"


def _capture_original_dtlz6_pool():
    saved=front.decision_vector
    decisions=[]
    def capture(candidate,**kwargs):
        value=saved(candidate,**kwargs)
        decisions.append(tuple(float(v) for v in value))
        return value
    try:
        front.decision_vector=capture
        original=np.asarray(DTLZ6(M=4,K=1).ps(n_points=16),dtype=float)
    finally:
        front.decision_vector=saved
    pool=np.asarray(decisions,dtype=float)
    if pool.shape!=(448,4) or original.shape!=(16,4):
        raise AssertionError("Original DTLZ6 source was not reconstructed")
    return original,pool


def _alternative_dtlz6(original,pool,eligible):
    # Match the independently published original-source counterfactual:
    # same 4 anchors, same full pool+anchor normalization and maximin.
    all_y,_=_original_objectives(pool,problem="DTLZ6",m=4,k=1)
    anchor_y,_=_original_objectives(original[:4],problem="DTLZ6",m=4,k=1)
    scales=np.maximum(np.ptp(np.vstack((anchor_y,all_y)),axis=0),1e-12)
    pool_eligible_y=all_y[eligible]
    nearest=np.min(np.sum(
        ((pool_eligible_y[:,None,:]-anchor_y[None,:,:])/scales)**2,
        axis=2),axis=1)
    picked=[]
    for _ in range(12):
        j=int(np.argmax(nearest))
        picked.append(int(eligible[j]))
        delta=np.sum(
            ((pool_eligible_y-pool_eligible_y[j])/scales)**2,axis=1)
        nearest=np.minimum(nearest,delta)
        nearest[j]=-1.
    return np.vstack((original[:4],pool[picked])),picked


def _fingerprint(decisions):
    return hashlib.sha256(b"".join(
        float(v).hex().encode("ascii")+b";"
        for row in np.asarray(decisions) for v in row
    )).hexdigest()


def _bound_certificates(decisions,problem):
    zero=certify_gzero_curve_cover(decisions,problem=problem)
    first=certify_full_radial_face_to_source(decisions,problem=problem)
    second=certify_second_surface_to_original_ps(decisions,problem=problem)
    boundaries={
        "complete_gzero_curve":_decode(zero["raw_euclidean_cover_upper"]),
        "first_full_radial_support_surface":_decode(
            first["full_radial_face_to_actual_source_l2_upper"]),
        "second_full_radial_support_surface":_decode(
            second["continuous_surface_to_actual_ps_l2_upper"]),
    }
    if any(bound<0 for bound in boundaries.values()):
        raise ArithmeticError("Invalid rigorous continuous cover upper")
    return {
        "source_decisions_sha256":_fingerprint(decisions),
        "per_subset_rigorous_raw_l2_upper":{
            name:_encode(value) for name,value in boundaries.items()},
        "union_rigorous_raw_l2_upper":_encode(max(boundaries.values())),
        "gzero_exact_anchor_certificate":zero,
        "first_support_surface_certificate":first,
        "second_support_surface_certificate":second,
        "declared_subsets_only":True,
        "entire_original_pf_coverage":"unverified",
        "indicator_suitability":"unverified",
    }


def compare_continuous_support_coverage():
    trace=trace_original_dtlz5_selection(include_candidate_pool=True)
    reach=evaluate_proof_reach(trace)
    chosen=compare_proof_gated_maximin(
        trace,reach,require_low_g=True)
    if chosen["status"]!="observational_proved_only_stratified_maximin":
        raise ArithmeticError("No proved-only stratified DTLZ5 alternative")
    original5=np.asarray([
        [float.fromhex(v) for v in row["decision_hex"]]
        for row in trace["selected_rows"]
    ],dtype=float)
    pool5=np.asarray([
        [float.fromhex(v) for v in row]
        for row in trace["candidate_pool_decision_hex"]
    ],dtype=float)
    indices5=chosen["proved_source_candidate_indices"]
    alternative5=np.vstack((original5[:4],pool5[indices5]))
    if any(reach["candidate_rows"][i]["status"]!="proved_pf"
           for i in indices5):
        raise AssertionError("DTLZ5 alternative has unproved point")

    old6,pool6=_capture_original_dtlz6_pool()
    record6=audit_dtlz6_source()
    eligible6=np.asarray([
        row["index"] for row in record6["candidate_rows"]
        if row["status"]=="proved_pf"
    ],dtype=int)
    alternative6,indices6=_alternative_dtlz6(old6,pool6,eligible6)
    historical6=record6["counterfactual"]["selected_proved_candidate_indices"]
    if indices6!=historical6:
        raise AssertionError("DTLZ6 original proof-gated selection drift")
    if len(indices5)!=12 or len(indices6)!=12:
        raise AssertionError("Wrong original 16-point selection cardinality")
    if not np.array_equal(original5[:4],alternative5[:4]) or (
        not np.array_equal(old6[:4],alternative6[:4])
    ):
        raise AssertionError("Original four g0 anchors not preserved")
    result={}
    for problem,old,new,picked in (
        ("DTLZ5",original5,alternative5,indices5),
        ("DTLZ6",old6,alternative6,indices6),
    ):
        before=_bound_certificates(old,problem)
        after=_bound_certificates(new,problem)
        if before["source_decisions_sha256"]==after["source_decisions_sha256"]:
            raise AssertionError("Counterfactual did not alter original selection")
        result[problem]={
            "original_selected_sha256":_fingerprint(old),
            "alternative_selected_sha256":_fingerprint(new),
            "same_first_four_gzero_anchors":True,
            "proof_gated_pool_indices":list(picked),
            "original_declared_subset_coverage":before,
            "alternative_declared_subset_coverage":after,
            "scientifically_known_bad_original_selected_indices":(
                [10] if problem=="DTLZ5" else [7,11]),
            "alternative_individual_membership":
                "proved_for_all_16_under_existing_exact_oracles",
            "global_original_pf_exhaustiveness":"unverified",
            "full_reference_IGD_IGDplus_HV_acceptance":"unverified",
        }
    return {
        "schema":SCHEMA,"configuration":{"M":4,"K":1,"n_points":16,
            "original_scalarization_pool":448},
        "DTLZ5":result["DTLZ5"],"DTLZ6":result["DTLZ6"],
        "bounds_are_upper_not_exact_distances":True,
        "complete_original_pareto_front":"unverified",
        "release_scientific_acceptance":"unverified",
        "production_generator_modified":False,
        "accepted_adr_modified":False,
        "existing_regressions_modified":False,
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    a=parser.parse_args(argv)
    if a.report.exists():
        raise FileExistsError("Research source-bound report is write-once")
    result=compare_continuous_support_coverage()
    with a.report.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    for problem in ("DTLZ5","DTLZ6"):
        entry=result[problem]
        for name in ("original_declared_subset_coverage",
                     "alternative_declared_subset_coverage"):
            evidence=entry[name]
            print(problem,name,"rigorous per-component raw L2 upper",
                  {k:float(_decode(v)) for k,v in
                   evidence["per_subset_rigorous_raw_l2_upper"].items()})
            print(problem,name,"rigorous subset union raw L2 upper",
                  float(_decode(evidence["union_rigorous_raw_l2_upper"])))
        print(problem,"entire original PF: UNVERIFIED")


if __name__=="__main__":
    main()
