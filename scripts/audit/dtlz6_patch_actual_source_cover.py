# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rigorous DTLZ6 positive-g continuous PATCH-to-ACTUAL-source bound.

Original M=4 K=1 and exact binary selected source decisions. Exact
rational tenth-root outward enclosures give intervals [glo,ghi].
Inner spherical angle coefficient
  c(g,x)=(1+2*g*x)/(4*(1+g))
is monotonic in g, so exact min/max occur at interval endpoints.
The pi/trigonometric intervals are rigorously outward via rational
Machin and Taylor bounds, nonnegative-clamped at endpoint zeros.
Combining interval-wise L1 objective distances and the previously
proved positive-g patch-to-synthetic-grid Lipschitz upper bound
gives a continuous restricted patch-to-ACTUAL-selected-PS theorem.
NOT full PF/GT/indicator certification or stored finite F equality.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from .dtlz6_positive_g_patch_cover import (
    certify_dtlz6_positive_g_patch_cover,
    replay_dtlz6_positive_g_patch_cover,
)
from .dtlz56_dtlz6_radial_enclosure import rational_tenth_root_enclosure
from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz5_row10_exact_dominance import _trig_bounds
from .dtlz5_patch_actual_source_cover import (
    _decode,_exact_decisions,_interval_l1_distance_upper,
)
from .dtlz56_exact_same_ray import _encode

PROOF_ID="DTLZ6-M4K1-PATCH-TO-ACTUAL-PS-EXACT-UPPER-V1"


def _outward_original_objectives(decision,pi_bounds,*,root_bits):
    x1,x2,x3,tail=decision
    glo,ghi=rational_tenth_root_enclosure(tail,bits=root_bits)
    if not 0<=glo<=ghi<=1:
        raise ArithmeticError("Invalid outward original tenth-root interval")

    def angular_coeff_interval(x):
        def c(g):
            return (1+2*g*x)/(4*(1+g))
        left,right=c(glo),c(ghi)
        return min(left,right),max(left,right)

    bounds=((x1/2,x1/2),
            angular_coeff_interval(x2),
            angular_coeff_interval(x3))
    cosines=[]
    sines=[]
    for lo,hi in bounds:
        if not 0<=lo<=hi<=Fraction(1,2):
            raise ArithmeticError("Invalid original angular bounds")
        cos_lo=_trig_bounds(hi,pi_bounds,sine=False)[0]
        cos_hi=_trig_bounds(lo,pi_bounds,sine=False)[1]
        sin_lo=_trig_bounds(lo,pi_bounds,sine=True)[0]
        sin_hi=_trig_bounds(hi,pi_bounds,sine=True)[1]
        cosines.append((max(Fraction(0),cos_lo),cos_hi))
        sines.append((max(Fraction(0),sin_lo),sin_hi))

    products=(
        (cosines[0],cosines[1],cosines[2]),
        (cosines[0],cosines[1],sines[2]),
        (cosines[0],sines[1]),
        (sines[0],),
    )
    output=[]
    for group in products:
        a,b=1+glo,1+ghi
        for lo,hi in group:
            if not 0<=lo<=hi:
                raise ArithmeticError("Invalid outward factor interval")
            a*=lo
            b*=hi
        output.append((a,b))
    return output


def certify_dtlz6_patch_to_source_cover(
    decisions, *, x1_grid=(0.,1/16,1/8),
    u_grid=(0.,1/8,1/4), root_bits=56, pi_terms=16,
):
    ps=_exact_decisions(decisions)
    if type(root_bits) is not int or not 8<=root_bits<=80:
        raise ValueError("Invalid exact tenth-root enclosure precision")
    patch=certify_dtlz6_positive_g_patch_cover(
        x1_grid=x1_grid,u_grid=u_grid,pi_terms=pi_terms)
    if not replay_dtlz6_positive_g_patch_cover(
        patch,x1_grid=x1_grid,u_grid=u_grid,pi_terms=pi_terms):
        raise ArithmeticError("Original DTLZ6 patch theorem did not replay")

    pi_bounds=_pi_rational_bounds(pi_terms)
    intervals=[
        _outward_original_objectives(row,pi_bounds,root_bits=root_bits)
        for row in ps
    ]
    results=[]
    for node in patch["grid_decision_images"]:
        decision=tuple(_decode(v) for v in node["decision"])
        grid_intervals=_outward_original_objectives(
            decision,pi_bounds,root_bits=root_bits)
        distances=[
            _interval_l1_distance_upper(grid_intervals,r)
            for r in intervals
        ]
        chosen=min(range(len(ps)),key=lambda j:(distances[j],j))
        results.append({
            "grid_decision":node["decision"],
            "nearest_source_index":chosen,
            "objective_l1_upper":_encode(distances[chosen]),
        })
    maximum=max(_decode(item["objective_l1_upper"]) for item in results)
    patch_bound=_decode(patch["raw_euclidean_continuous_patch_cover_upper"])
    upper=patch_bound+maximum
    return {
        "proof_id":PROOF_ID,
        "problem":"DTLZ6","M":4,"K":1,
        "conclusion":"rigorously_bounded_continuous_patch_to_selected_source",
        "selected_decision_count":len(ps),
        "selected_decisions":[[_encode(x) for x in row] for row in ps],
        "selected_decisions_sha256":hashlib.sha256(
            b"".join(float(x).hex().encode("ascii")+b";"
                     for row in ps for x in row)
        ).hexdigest(),
        "root_bits":root_bits,"pi_terms":pi_terms,
        "patch_certificate":patch,
        "per_grid_node_nearest_actual_source":results,
        "worst_grid_to_source_l1_upper":_encode(maximum),
        "continuous_patch_to_grid_l2_upper":
            patch["raw_euclidean_continuous_patch_cover_upper"],
        "continuous_patch_to_actual_source_l2_upper":_encode(upper),
        "complete_positive_g_pf_coverage":"unverified",
        "finite_precision_objective_F_alignment":"unverified",
        "whole_gt_coverage":"unverified",
        "scientific_certification":"unverified",
    }


def replay_dtlz6_patch_to_source_cover(
    decisions, record, *, x1_grid=(0.,1/16,1/8),
    u_grid=(0.,1/8,1/4),root_bits=56,pi_terms=16,
):
    try:
        proof=certify_dtlz6_patch_to_source_cover(
            decisions,x1_grid=x1_grid,u_grid=u_grid,
            root_bits=root_bits,pi_terms=pi_terms)
        return (record==proof and
                proof["conclusion"]==
                "rigorously_bounded_continuous_patch_to_selected_source")
    except (ValueError,TypeError,ArithmeticError,OverflowError,
            ZeroDivisionError):
        return False


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-bundle",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Research evidence cannot be overwritten")
    digest=hashlib.sha256(args.paired_bundle.read_bytes()).hexdigest()
    with np.load(args.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected paired original source members")
        if (str(data["problem"].item())!="DTLZ6" or
            int(data["M"].item())!=4 or int(data["K"].item())!=1):
            raise ValueError("Wrong original paired problem/dimensions")
        ps=data["decisions"].copy()
    proof=certify_dtlz6_patch_to_source_cover(ps)
    if not replay_dtlz6_patch_to_source_cover(ps,proof):
        raise ArithmeticError("Actual selected source proof replay failed")
    proof["paired_artifact_sha256"]=digest
    with args.report.open("x",encoding="utf-8") as stream:
        json.dump(proof,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    upper=_decode(proof["continuous_patch_to_actual_source_l2_upper"])
    print("Actual selected original DTLZ6 patch cover upper:",float(upper))
    print("Paired PS/F SHA256:",digest)
    print("Whole GT certificate:",proof["scientific_certification"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
