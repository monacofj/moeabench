# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Proof-carrying union covering bound for 3 certified GT components.

Original DTLZ5/6 M=4 K=1 only. Input: the SAME immutable actual paired
selected PS/F archive, the full-radial support-face source certificate,
and second-support curve source certificate. Independently REPLAY both
source-bound certificates against the exact paired PS bytes; recompute
the g=0 continuous curve source certificate directly. A finite union of
covered sets has covering radius at most the maximum of component-wise
radii. This is a mathematically rigorous raw L2 bound for the exact
union of three declared continuous Pareto subsets, NOT whole Pareto
front completeness, stored finite F equality or indicator acceptance.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from .dtlz56_gzero_continuous_cover import (
    certify_gzero_curve_cover,replay_gzero_curve_cover,
)
from .dtlz56_full_radial_face_actual_source import (
    replay_full_radial_face_to_source,
)
from .dtlz56_second_support_actual_source import (
    replay_second_support_to_actual_ps,
)
from .dtlz5_patch_actual_source_cover import _decode
from .dtlz56_exact_same_ray import _encode

SCHEMA="dtlz56_m4k1_three_continuous_pf_subsets_to_selected_ps_v1"


def certify_three_subset_union(
    decisions, *,problem,full_radial_report,second_support_report,
):
    if problem not in ("DTLZ5","DTLZ6"):
        raise ValueError("Only original DTLZ5/6 M4 K1")
    if not replay_full_radial_face_to_source(
        decisions,full_radial_report,problem=problem):
        raise ValueError("Full radial original source proof failed replay")
    if not replay_second_support_to_actual_ps(
        decisions,second_support_report,problem=problem):
        raise ValueError("Second global support original source proof failed replay")
    zero=certify_gzero_curve_cover(decisions,problem=problem)
    if not replay_gzero_curve_cover(decisions,zero,problem=problem):
        raise ValueError("Actual source has no certifiable full g=0 anchors")
    if not (full_radial_report["source_decisions"]==
            second_support_report["selected_decisions"]):
        raise ValueError("Source decision statements disagree")
    if not (full_radial_report["source_decisions_sha256"]==
            second_support_report["selected_decisions_sha256"]):
        raise ValueError("Canonical source decision digest mismatch")
    bounds={
        "complete_gzero_continuous_curve":_decode(
            zero["raw_euclidean_cover_upper"]),
        "complete_radial_first_support_face":_decode(
            full_radial_report["full_radial_face_to_actual_source_l2_upper"]),
        "complete_radial_second_support_curve":_decode(
            second_support_report["continuous_curve_to_actual_source_l2_upper"]),
    }
    upper=max(bounds.values())
    return {
        "schema":SCHEMA,"problem":problem,"M":4,"K":1,
        "conclusion":"rigorously_bounded_three_declared_continuous_pf_subsets",
        "actual_selected_decision_count":len(decisions),
        "selected_decisions_sha256":
            full_radial_report["source_decisions_sha256"],
        "gzero_exact_certificate":zero,
        "continuous_component_upper_bounds":{
            name:_encode(value) for name,value in bounds.items()
        },
        "union_continuous_cover_raw_l2_upper":_encode(upper),
        "union_construction":"finite union: max of three rigorous one-sided bounds",
        "certified_subset_count":3,
        "full_positive_g_pf_component_completeness":"unverified",
        "whole_pareto_front_cover":"unverified",
        "finite_stored_F_exact_alignment":"unverified",
        "igd_igdplus_hv_suitability":"unverified",
        "scientific_certification":"unverified",
    }


def replay_three_subset_union(
    decisions,record,*,problem,full_radial_report,second_support_report,
):
    try:
        correct=certify_three_subset_union(
            decisions,problem=problem,full_radial_report=full_radial_report,
            second_support_report=second_support_report)
        return (record==correct and
                correct["conclusion"]==
                "rigorously_bounded_three_declared_continuous_pf_subsets")
    except (TypeError,ValueError,KeyError,AssertionError,
            ArithmeticError,OverflowError,ZeroDivisionError):
        return False


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-bundle",required=True,type=Path)
    parser.add_argument("--full-radial",required=True,type=Path)
    parser.add_argument("--second-support",required=True,type=Path)
    parser.add_argument("--report",required=True,type=Path)
    args=parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Research evidence must be immutable")
    digest=hashlib.sha256(args.paired_bundle.read_bytes()).hexdigest()
    with np.load(args.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected paired source file schema")
        problem=str(data["problem"].item())
        if problem not in ("DTLZ5","DTLZ6") or (
            int(data["M"].item())!=4 or int(data["K"].item())!=1
        ):
            raise ValueError("Unexpected original problem / dimension")
        decisions=data["decisions"].copy()
    full=json.loads(args.full_radial.read_text(encoding="utf-8"))
    second=json.loads(args.second_support.read_text(encoding="utf-8"))
    if full.get("paired_artifact_sha256")!=digest or (
        second.get("paired_artifact_sha256")!=digest
    ):
        raise ValueError("Proof records not bound to identical paired artifact")
    result=certify_three_subset_union(
        decisions,problem=problem,full_radial_report=full,
        second_support_report=second)
    if not replay_three_subset_union(
        decisions,result,problem=problem,full_radial_report=full,
        second_support_report=second):
        raise ArithmeticError("Independent union certificate failed replay")
    result["paired_artifact_sha256"]=digest
    with args.report.open("x",encoding="utf-8") as fp:
        json.dump(result,fp,sort_keys=True,indent=2,allow_nan=False)
        fp.write("\n")
    print(problem,"certified THREE-subset union to ACTUAL original PS raw L2 upper:",
          float(_decode(result["union_continuous_cover_raw_l2_upper"])))
    print(problem,"subsets:",
          {key:float(_decode(value)) for key,value
           in result["continuous_component_upper_bounds"].items()})
    print(problem,"scientific certification:",result["scientific_certification"])


if __name__=="__main__":
    main()
