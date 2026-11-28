# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact positive radial FULL-RANGE support-face cover by ACTUAL original PS.

The source-bound original DTLZ5/6 M4/K1 selected decisions are held
unchanged. The continuous x2=x3=1 face at all admissible radial values
is globally Pareto optimal by the separate uniform exact theorem.

For every synthetic grid node on that face, rigorous rational outward
original-equation intervals produce an L1 upper bound to each actual
selected source decision image. Combining the worst nearest upper with
the full-face continuous-to-grid Lipschitz bound yields a rigorous
(one-sided, raw L2) face-to-ACTUAL original PS upper bound.

This does not cover the remaining angular PF strata or certify the
full GT; a dominated source point remains a scientific rejection.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from .dtlz56_full_radial_support_face import (
    certify_full_radial_support_face,replay_full_radial_support_face,
)
from .dtlz5_patch_actual_source_cover import (
    _decode,_exact_decisions,_interval_l1_distance_upper,
    _endpoint_safe_objective_intervals,
)
from .dtlz6_patch_actual_source_cover import _outward_original_objectives
from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz56_exact_same_ray import _encode

PROOF_ID="DTLZ56-M4K1-FULL-RADIAL-FACE-ACTUAL-SOURCE-COVER-V1"


def certify_full_radial_face_to_source(
    decisions, *, problem,pi_terms=16,root_bits=56,
):
    ps=_exact_decisions(decisions)
    face=certify_full_radial_support_face(
        problem=problem,pi_terms=pi_terms)
    if not replay_full_radial_support_face(
        face,problem=problem,pi_terms=pi_terms):
        raise ArithmeticError("Independent full-radial face theorem failed")
    pi=_pi_rational_bounds(pi_terms)
    if problem=="DTLZ5":
        def equation(row):
            return _endpoint_safe_objective_intervals(row,pi)
    else:
        def equation(row):
            return _outward_original_objectives(
                row,pi,root_bits=root_bits)
    ps_intervals=[equation(row) for row in ps]
    nodes=[]
    for item in face["grid_decisions"]:
        decision=tuple(_decode(v) for v in item["decision"])
        f=equation(decision)
        distances=[
            _interval_l1_distance_upper(f,p) for p in ps_intervals
        ]
        j=min(range(len(ps)),key=lambda index:(distances[index],index))
        nodes.append({
            "grid_decision":item["decision"],
            "nearest_actual_source_index":j,
            "objective_l1_upper":_encode(distances[j]),
        })
    worst=max(_decode(n["objective_l1_upper"]) for n in nodes)
    surface=_decode(face["continuous_to_exact_grid_raw_l2_upper"])
    upper=worst+surface
    return {
        "proof_id":PROOF_ID,"problem":problem,"M":4,"K":1,
        "conclusion":"rigorous_full_radial_face_to_actual_source_bound",
        "source_decision_count":len(ps),
        "source_decisions":[[_encode(v) for v in row] for row in ps],
        "source_decisions_sha256":hashlib.sha256(
            b"".join(float(v).hex().encode("ascii")+b";"
                     for row in ps for v in row)
        ).hexdigest(),
        "uniform_full_radial_face_pf_certificate":face,
        "grid_nearest_actual_source":nodes,
        "continuous_to_grid_l2_upper":_encode(surface),
        "worst_grid_to_source_l1_upper":_encode(worst),
        "full_radial_face_to_actual_source_l2_upper":_encode(upper),
        "pi_terms":pi_terms,"root_bits":root_bits,
        "remaining_angular_support_faces":"unverified",
        "finite_stored_F_exactness":"unverified",
        "whole_pf_coverage":"unverified",
        "indicator_fitness":"unverified",
        "scientific_certification":"unverified",
    }


def replay_full_radial_face_to_source(
    decisions,record,*,problem,pi_terms=16,root_bits=56,
):
    try:
        expected=certify_full_radial_face_to_source(
            decisions,problem=problem,pi_terms=pi_terms,root_bits=root_bits)
        return (record==expected and expected["conclusion"]==
                "rigorous_full_radial_face_to_actual_source_bound")
    except (ValueError,TypeError,ArithmeticError,AssertionError,
            ZeroDivisionError,OverflowError):
        return False


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-bundle",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Research source proof cannot be overwritten")
    digest=hashlib.sha256(args.paired_bundle.read_bytes()).hexdigest()
    with np.load(args.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected paired source bundle layout")
        problem=str(data["problem"].item())
        if problem not in ("DTLZ5","DTLZ6"):
            raise ValueError("Unexpected original problem")
        if int(data["M"])!=4 or int(data["K"])!=1:
            raise ValueError("Unexpected source dimensions")
        decisions=data["decisions"].copy()
    r=certify_full_radial_face_to_source(decisions,problem=problem)
    if not replay_full_radial_face_to_source(
        decisions,r,problem=problem):
        raise ArithmeticError("Source-bound exact replay failed")
    r["paired_artifact_sha256"]=digest
    with args.report.open("x",encoding="utf-8") as fp:
        json.dump(r,fp,sort_keys=True,indent=2,allow_nan=False)
        fp.write("\n")
    print(problem,"rigorous full radial PF support FACE to actual source upper",
          float(_decode(r["full_radial_face_to_actual_source_l2_upper"])))
    print(problem,"continuum-to-grid",float(_decode(
        r["continuous_to_grid_l2_upper"])))
    print(problem,"grid-to-actual",float(_decode(
        r["worst_grid_to_source_l1_upper"])))
    print(problem,"paired SHA256",digest)
    print(problem,"whole GT certification",r["scientific_certification"])


if __name__=="__main__":
    main()
