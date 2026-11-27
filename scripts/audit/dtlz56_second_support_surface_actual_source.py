# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rigorous second-support 2D full-radial surface coverage by ORIGINAL PS.

The independent mathematical PF theorem certifies every real point
of x2=1,x3=0, 0<=x1<=xmax and the full feasible radial range.
For each declared exact grid node, outward rational original-equation
intervals rigorously bound raw L1 distances to each unchanged selected
binary source decision image (DTLZ6 includes tenth-root enclosures).
A continuous spherical Lipschitz bound plus the worst grid-node
nearest-reference L1 upper yields the surface-to-ACTUAL-PS raw L2 bound.

No claim about all angular Pareto strata or stored rounded F.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np

from .dtlz56_second_support_full_radial_surface import (
    certify_second_support_full_radial_surface,
    replay_second_support_full_radial_surface,
)
from .dtlz5_patch_actual_source_cover import (
    _decode,_exact_decisions,_interval_l1_distance_upper,
    _endpoint_safe_objective_intervals,
)
from .dtlz6_patch_actual_source_cover import _outward_original_objectives
from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz56_exact_same_ray import _encode

SCHEMA="dtlz56_m4k1_second_support_surface_actual_source_v1"


def certify_second_surface_to_original_ps(
    decisions, *,problem,pi_terms=16,root_bits=56,
):
    ps=_exact_decisions(decisions)
    surface=certify_second_support_full_radial_surface(
        problem=problem,pi_terms=pi_terms)
    if not replay_second_support_full_radial_surface(
        surface,problem=problem,pi_terms=pi_terms):
        raise ArithmeticError("Second-support continuous PF surface proof failed")
    pi=_pi_rational_bounds(pi_terms)
    if problem=="DTLZ5":
        def interval(row):
            return _endpoint_safe_objective_intervals(row,pi)
    else:
        def interval(row):
            return _outward_original_objectives(
                row,pi,root_bits=root_bits)
    selected=[interval(row) for row in ps]
    node_evidence=[]
    for node in surface["grid_decisions"]:
        q=tuple(_decode(v) for v in node["decision"])
        qi=interval(q)
        distances=[_interval_l1_distance_upper(qi,v) for v in selected]
        j=min(range(len(distances)),key=lambda x:(distances[x],x))
        node_evidence.append({
            "grid_decision":node["decision"],
            "nearest_actual_ps_index":j,
            "rigorous_original_objective_l1_upper":_encode(distances[j]),
        })
    worst=max(_decode(p["rigorous_original_objective_l1_upper"])
              for p in node_evidence)
    smooth=_decode(surface["continuous_to_grid_raw_l2_upper"])
    return {
        "schema":SCHEMA,"problem":problem,"M":4,"K":1,
        "conclusion":"rigorous_second_support_surface_to_actual_original_ps",
        "source_decision_count":len(ps),
        "source_decisions":[[_encode(v) for v in row] for row in ps],
        "source_decisions_sha256":hashlib.sha256(
            b"".join(float(x).hex().encode("ascii")+b";"
                     for row in ps for x in row)
        ).hexdigest(),
        "independent_continuous_surface_pf_proof":surface,
        "grid_to_actual_source_proof_nodes":node_evidence,
        "continuum_to_grid_l2_upper":_encode(smooth),
        "grid_to_selected_source_l1_upper":_encode(worst),
        "continuous_surface_to_actual_ps_l2_upper":_encode(smooth+worst),
        "pi_terms":pi_terms,"root_bits":root_bits,
        "full_positive_g_pf_coverage":"unverified",
        "stored_finite_F_exactness":"unverified",
        "indicator_quality":"unverified",
        "scientific_certification":"unverified",
    }


def replay_second_surface_to_original_ps(
    decisions,record,*,problem,pi_terms=16,root_bits=56,
):
    try:
        correct=certify_second_surface_to_original_ps(
            decisions,problem=problem,pi_terms=pi_terms,root_bits=root_bits)
        return (record==correct and correct["conclusion"]==
                "rigorous_second_support_surface_to_actual_original_ps")
    except (TypeError,ValueError,KeyError,ArithmeticError,
            AssertionError,ZeroDivisionError,OverflowError):
        return False


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-bundle",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Research certificates cannot be overwritten")
    digest=hashlib.sha256(args.paired_bundle.read_bytes()).hexdigest()
    with np.load(args.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected paired original GT archive")
        problem=str(data["problem"].item())
        if problem not in ("DTLZ5","DTLZ6") or (
            int(data["M"].item())!=4 or int(data["K"].item())!=1
        ):
            raise ValueError("Incorrect original source problem or dimensions")
        decisions=data["decisions"].copy()
    proof=certify_second_surface_to_original_ps(decisions,problem=problem)
    if not replay_second_surface_to_original_ps(
        decisions,proof,problem=problem):
        raise ArithmeticError("Independent exact source proof replay failed")
    proof["paired_artifact_sha256"]=digest
    with args.report.open("x",encoding="utf-8") as output:
        json.dump(proof,output,sort_keys=True,indent=2,allow_nan=False)
        output.write("\n")
    print(problem,"second full-radial SUPPORT SURFACE actual PS L2 upper:",
          float(_decode(proof["continuous_surface_to_actual_ps_l2_upper"])))
    print(problem,"original paired artifact SHA256",digest)
    print(problem,"whole-GT certificate:",proof["scientific_certification"])


if __name__=="__main__":
    main()
