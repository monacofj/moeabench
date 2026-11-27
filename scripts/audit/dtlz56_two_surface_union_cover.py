# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Proof-carrying coverage union of g=0 and TWO full-radial PF surfaces.

The previous three-subset union contained a second-support 1D
radial curve; the new second-support 2D surface strictly includes
that curve. Replay the previous source-bound three-subset union
and the new second-surface source-bound certificate against the
same unchanged original selected PS. Then take the maximum of:
  complete g=0 curve, first complete radial support surface,
  second complete radial support surface.
This is rigorous for the union of declared continuous subsets,
NOT a complete Pareto-front decomposition or finite GT pass.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np

from .dtlz56_three_subset_union_cover import (
    certify_three_subset_union,replay_three_subset_union,
)
from .dtlz56_second_support_surface_actual_source import (
    replay_second_surface_to_original_ps,
)
from .dtlz5_patch_actual_source_cover import _decode
from .dtlz56_exact_same_ray import _encode

SCHEMA="dtlz56_m4k1_two_full_radial_surfaces_union_to_original_ps_v1"


def certify_two_surface_union(
    decisions,*,problem,first_report,second_curve_report,
    second_surface_report,
):
    if not replay_second_surface_to_original_ps(
        decisions,{k:v for k,v in second_surface_report.items()
                   if k!="paired_artifact_sha256"},problem=problem):
        raise ValueError("Second full-radial SURFACE proof did not replay")
    base=certify_three_subset_union(
        decisions,problem=problem,full_radial_report=first_report,
        second_support_report=second_curve_report)
    if not replay_three_subset_union(
        decisions,base,problem=problem,full_radial_report=first_report,
        second_support_report=second_curve_report):
        raise ValueError("Existing continuous original PF union did not replay")
    if not (
        base["selected_decisions_sha256"]==
        second_surface_report["source_decisions_sha256"] and
        first_report["source_decisions"]==
        second_surface_report["source_decisions"]
    ):
        raise ValueError("Original PS source canonical decision identity mismatch")
    b={
        "complete_gzero_curve":_decode(
            base["continuous_component_upper_bounds"][
                "complete_gzero_continuous_curve"]),
        "complete_radial_first_support_surface":_decode(
            base["continuous_component_upper_bounds"][
                "complete_radial_first_support_face"]),
        "complete_radial_second_support_surface":_decode(
            second_surface_report["continuous_surface_to_actual_ps_l2_upper"]),
    }
    maximum=max(b.values())
    return {
        "schema":SCHEMA,"problem":problem,"M":4,"K":1,
        "conclusion":"rigorous_two_full_radial_surface_union_cover",
        "original_selected_n":len(decisions),
        "selected_decisions_sha256":base["selected_decisions_sha256"],
        "component_bounds":{k:_encode(v) for k,v in b.items()},
        "union_continuous_raw_l2_upper":_encode(maximum),
        "source_verified_three_subset_ledger":base,
        "second_surface_source_proof_sha256":hashlib.sha256(
            json.dumps(second_surface_report,sort_keys=True,
                       separators=(",",":")).encode()
        ).hexdigest(),
        "second_support_radial_curve_included_in_second_surface":True,
        "continuous_declared_subset_only":True,
        "complete_original_pf_decomposition":"unverified",
        "whole_gt_coverage":"unverified",
        "igd_igdplus_hv_quality":"unverified",
        "scientific_certification":"unverified",
    }


def replay_two_surface_union(
    decisions,record,*,problem,first_report,second_curve_report,
    second_surface_report,
):
    try:
        expected=certify_two_surface_union(
            decisions,problem=problem,first_report=first_report,
            second_curve_report=second_curve_report,
            second_surface_report=second_surface_report)
        return (record==expected and expected["conclusion"]==
                "rigorous_two_full_radial_surface_union_cover")
    except (TypeError,ValueError,KeyError,ArithmeticError,
            AssertionError,ZeroDivisionError,OverflowError):
        return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-bundle",required=True,type=Path)
    p.add_argument("--first-support",required=True,type=Path)
    p.add_argument("--second-curve",required=True,type=Path)
    p.add_argument("--second-surface",required=True,type=Path)
    p.add_argument("--report",required=True,type=Path)
    a=p.parse_args(argv)
    if a.report.exists():
        raise FileExistsError("Research proof output cannot be overwritten")
    digest=hashlib.sha256(a.paired_bundle.read_bytes()).hexdigest()
    with np.load(a.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Incorrect paired original source schema")
        problem=str(data["problem"].item())
        if problem not in ("DTLZ5","DTLZ6") or (
            int(data["M"].item())!=4 or int(data["K"].item())!=1
        ):
            raise ValueError("Unexpected original problem/dimension")
        ps=data["decisions"].copy()
    first=json.loads(a.first_support.read_text(encoding="utf-8"))
    curve=json.loads(a.second_curve.read_text(encoding="utf-8"))
    surface=json.loads(a.second_surface.read_text(encoding="utf-8"))
    for item in (first,curve,surface):
        if item.get("paired_artifact_sha256")!=digest:
            raise ValueError("Source proof does not identify same paired PS/F archive")
    proof=certify_two_surface_union(
        ps,problem=problem,first_report=first,
        second_curve_report=curve,second_surface_report=surface)
    if not replay_two_surface_union(
        ps,proof,problem=problem,first_report=first,
        second_curve_report=curve,second_surface_report=surface):
        raise ArithmeticError("Independent whole proof replay failed")
    proof["paired_artifact_sha256"]=digest
    with a.report.open("x",encoding="utf-8") as out:
        json.dump(proof,out,sort_keys=True,indent=2,allow_nan=False)
        out.write("\n")
    print(problem,"rigorous 2 FULL RADIAL SURFACE union original PS upper:",
          float(_decode(proof["union_continuous_raw_l2_upper"])))
    print(problem,"component uppers:",
          {key:float(_decode(value))
           for key,value in proof["component_bounds"].items()})
    print(problem,"scientific certification:",proof["scientific_certification"])


if __name__=="__main__":
    main()
