# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact continuous second support curve coverage by actual original selected PS.

For original DTLZ5 or DTLZ6 M=4,K=1, every decision on
(x1,x2,x3)=(0,1,0) with complete admissible radial range has an
independent global Pareto proof. Combine its exact-rational
continuum-to-grid Lipschitz bound with exact outward objective intervals
from real source-selected binary decisions. The maximum of grid-node
nearest-source *rigorous L1* uppers plus the continuous-to-grid
*rational L2* upper is a rigorous curve-to-real-PS L2 upper bound.

No whole-PF, full GT, stored F precision or indicator pass is implied.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from .dtlz56_second_support_full_radial import (
    certify_second_support_full_radial,replay_second_support_full_radial,
)
from .dtlz5_patch_actual_source_cover import (
    _decode,_exact_decisions,_interval_l1_distance_upper,
    _endpoint_safe_objective_intervals,
)
from .dtlz6_patch_actual_source_cover import _outward_original_objectives
from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz56_exact_same_ray import _encode

SCHEMA="dtlz56_second_support_full_radial_actual_ps_v1"


def certify_second_support_to_actual_ps(
    decisions, *,problem, pi_terms=16,root_bits=56,
):
    ps=_exact_decisions(decisions)
    proof=certify_second_support_full_radial(problem=problem,pi_terms=pi_terms)
    if not replay_second_support_full_radial(
        proof,problem=problem,pi_terms=pi_terms):
        raise ArithmeticError("Independent positive stratum proof did not replay")
    pi=_pi_rational_bounds(pi_terms)
    if problem=="DTLZ5":
        def interval(row):
            return _endpoint_safe_objective_intervals(row,pi)
    else:
        def interval(row):
            return _outward_original_objectives(row,pi,root_bits=root_bits)
    chosen_intervals=[interval(row) for row in ps]
    nodes=[]
    for grid in proof["grid_decisions"]:
        point=tuple(_decode(x) for x in grid["decision"])
        p_interval=interval(point)
        bounds=[
            _interval_l1_distance_upper(p_interval,x)
            for x in chosen_intervals
        ]
        j=min(range(len(ps)),key=lambda i:(bounds[i],i))
        nodes.append({
            "grid_decision":grid["decision"],
            "nearest_source_index":j,
            "rigorous_source_l1_upper":_encode(bounds[j]),
        })
    worst=max(_decode(x["rigorous_source_l1_upper"]) for x in nodes)
    interpolation=_decode(proof["raw_l2_continuous_to_grid_upper"])
    return {
        "schema":SCHEMA,"problem":problem,"M":4,"K":1,
        "conclusion":"rigorous_second_support_curve_to_actual_source_upper",
        "selected_decision_count":len(ps),
        "selected_decisions":[[_encode(v) for v in row] for row in ps],
        "selected_decisions_sha256":hashlib.sha256(
            b"".join(float(v).hex().encode("ascii")+b";"
                     for row in ps for v in row)
        ).hexdigest(),
        "curve_pf_proof":proof,
        "nearest_source_grid_nodes":nodes,
        "continuum_to_grid_l2_upper":_encode(interpolation),
        "grid_to_actual_source_l1_upper":_encode(worst),
        "continuous_curve_to_actual_source_l2_upper":_encode(interpolation+worst),
        "pi_terms":pi_terms,"root_bits":root_bits,
        "whole_pf_coverage":"unverified",
        "finite_F_alignment":"unverified",
        "indicator_fitness":"unverified",
        "scientific_certification":"unverified",
    }


def replay_second_support_to_actual_ps(
    decisions, record, *,problem,pi_terms=16,root_bits=56,
):
    try:
        expected=certify_second_support_to_actual_ps(
            decisions,problem=problem,pi_terms=pi_terms,root_bits=root_bits)
        return (record==expected and expected["conclusion"]==
                "rigorous_second_support_curve_to_actual_source_upper")
    except (TypeError,ValueError,ArithmeticError,AssertionError,
            ZeroDivisionError,OverflowError):
        return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-bundle",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    args=p.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Cannot overwrite archived mathematical result")
    sha=hashlib.sha256(args.paired_bundle.read_bytes()).hexdigest()
    with np.load(args.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected original source archive layout")
        problem=str(data["problem"].item())
        if problem not in ("DTLZ5","DTLZ6") or (
            int(data["M"])!=4 or int(data["K"])!=1
        ):
            raise ValueError("Unexpected problem or dimension")
        ps=data["decisions"].copy()
    result=certify_second_support_to_actual_ps(ps,problem=problem)
    if not replay_second_support_to_actual_ps(ps,result,problem=problem):
        raise ArithmeticError("Exact source certificate replay failed")
    result["paired_artifact_sha256"]=sha
    with args.report.open("x",encoding="utf-8") as file:
        json.dump(result,file,indent=2,sort_keys=True,allow_nan=False)
        file.write("\n")
    print(problem,"exact second-support full-radial curve-to-actual-PS upper",
          float(_decode(result["continuous_curve_to_actual_source_l2_upper"])))
    print(problem,"paired source SHA256",sha)
    return 0


if __name__=="__main__":
    main()
