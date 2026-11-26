# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rigorous DTLZ5 continuous positive-g PATCH-to-ACTUAL-source upper bound.

This external research companion takes the exact IEEE-754 binary decisions
of a real original M4/K1 selected PS. It combines:
 (1) independently replayed continuous-PF-patch-to-synthetic-grid
     Lipschitz upper bound,
 (2) at each exact synthetic grid node, a source decision whose
     mathematical objective-image distance is upper-bounded with
     outward exact rational trigonometric intervals.

By triangle inequality, the maximum over all patch points of nearest
real selected decision-image distance is bounded by the sum of (1) and
the maximum over grid nodes of (2). A conservative L1 norm is used in
(2) to avoid irrational square roots; it bounds Euclidean distance.

Every angle, pi enclosure and objective interval is exact Fraction.
This is a continuous restricted PATCH bound, NOT a whole positive-g PF,
not a statement about saved rounded objective F, and not a passing GT.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from .dtlz5_positive_g_patch_cover import (
    certify_positive_g_patch_cover,
    replay_positive_g_patch_cover,
)
from .dtlz5_row10_exact_dominance import _trig_bounds
from .dtlz56_rational_interval import _pi_rational_bounds
from .dtlz56_exact_same_ray import _encode

PROOF_ID="DTLZ5-M4K1-PATCH-TO-ACTUAL-PS-EXACT-UPPER-V1"


def _decode(record):
    return Fraction(int(record["numerator_hex"],16),
                    int(record["denominator_hex"],16))


def _exact_decisions(decisions):
    try:
        rows=tuple(tuple(Fraction.from_float(float(v)) for v in row)
                   for row in decisions)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Expected finite numeric original PS matrix") from exc
    if not rows or any(len(row)!=4 or any(not 0<=v<=1 for v in row)
                       for row in rows):
        raise ValueError("Original M4/K1 source PS must be nonempty and feasible")
    return rows


def _endpoint_safe_objective_intervals(decision, pi_bounds):
    """Conservative exact rational original equations even at cos(pi/2)=0.

    Taylor cosine lower polynomial may be negative near pi/2.
    Since original feasible sine/cosine factors are nonnegative,
    intersecting each rigorous interval with [0,+infinity) preserves
    containment, including exact endpoint zeros.
    """
    x1,x2,x3,tail=decision
    g=(tail-Fraction(1,2))**2
    r=1+g
    angular_coeffs=(
        x1/2,
        (1+2*g*x2)/(4*r),
        (1+2*g*x3)/(4*r),
    )
    if any(not 0<=c<=Fraction(1,2) for c in angular_coeffs):
        raise ValueError("Invalid original DTLZ5 angular parameter")
    cosines=[]
    sines=[]
    for coeff in angular_coeffs:
        clo,chi=_trig_bounds(coeff,pi_bounds,sine=False)
        slo,shi=_trig_bounds(coeff,pi_bounds,sine=True)
        cosines.append((max(Fraction(0),clo),chi))
        sines.append((max(Fraction(0),slo),shi))
    factors=(
        (cosines[0],cosines[1],cosines[2]),
        (cosines[0],cosines[1],sines[2]),
        (cosines[0],sines[1]),
        (sines[0],),
    )
    result=[]
    for group in factors:
        lower=upper=r
        for a,b in group:
            if not 0<=a<=b:
                raise ArithmeticError("Invalid outward nonnegative trig enclosure")
            lower*=a
            upper*=b
        result.append((lower,upper))
    return result


def _interval_l1_distance_upper(a,b):
    """Exact L1 bound for ANY values inside two outward interval vectors."""
    return sum((
        max(abs(xlo-yhi),abs(xhi-ylo))
        for (xlo,xhi),(ylo,yhi) in zip(a,b)
    ),Fraction(0))


def certify_dtlz5_patch_to_source_cover(
    decisions, *, x1_grid=(0.,1/16,1/8),
    tail_grid=(.5,.625,.75),pi_terms=16,
):
    """Exact-real upper bound to the *actual* supplied selected decisions."""
    ps=_exact_decisions(decisions)
    patch=certify_positive_g_patch_cover(
        x1_grid=x1_grid,tail_grid=tail_grid,pi_terms=pi_terms)
    if not replay_positive_g_patch_cover(
        patch,x1_grid=x1_grid,tail_grid=tail_grid,
        pi_terms=pi_terms):
        raise ArithmeticError("Continuous patch theorem did not replay")

    pi_bounds=_pi_rational_bounds(pi_terms)
    selected_intervals=[
        _endpoint_safe_objective_intervals(row,pi_bounds) for row in ps
    ]
    candidates=[]
    for node in patch["grid_decision_images"]:
        grid_row=tuple(_decode(v) for v in node["decision"])
        grid_intervals=_endpoint_safe_objective_intervals(
            grid_row,pi_bounds)
        distances=[
            _interval_l1_distance_upper(grid_intervals,source)
            for source in selected_intervals
        ]
        chosen=min(range(len(ps)),key=lambda j:(distances[j],j))
        candidates.append({
            "grid_decision":node["decision"],
            "nearest_source_index":chosen,
            "objective_l1_upper":_encode(distances[chosen]),
        })
    maximum=max(_decode(c["objective_l1_upper"]) for c in candidates)
    patch_upper=_decode(patch["raw_euclidean_continuous_patch_cover_upper"])
    total=patch_upper+maximum
    return {
        "proof_id":PROOF_ID,
        "problem":"DTLZ5","M":4,"K":1,
        "conclusion":"rigorously_bounded_continuous_patch_to_selected_source",
        "selected_decision_count":len(ps),
        "selected_decisions":[[_encode(v) for v in row] for row in ps],
        "selected_decisions_sha256":hashlib.sha256(
            b"".join(float(v).hex().encode("ascii")+b";"
                     for row in ps for v in row)
        ).hexdigest(),
        "patch_certificate":patch,
        "per_grid_node_nearest_actual_source":candidates,
        "worst_grid_to_source_l1_upper":_encode(maximum),
        "continuous_patch_to_grid_l2_upper":
            patch["raw_euclidean_continuous_patch_cover_upper"],
        "continuous_patch_to_actual_source_l2_upper":_encode(total),
        "pi_terms":pi_terms,
        "complete_positive_g_pf_coverage":"unverified",
        "finite_precision_objective_F_alignment":"unverified",
        "whole_gt_coverage":"unverified",
        "scientific_certification":"unverified",
    }


def replay_dtlz5_patch_to_source_cover(
    decisions, record, *,x1_grid=(0.,1/16,1/8),
    tail_grid=(.5,.625,.75),pi_terms=16,
):
    try:
        proof=certify_dtlz5_patch_to_source_cover(
            decisions,x1_grid=x1_grid,tail_grid=tail_grid,
            pi_terms=pi_terms)
        return (
            record==proof and proof["conclusion"]==
            "rigorously_bounded_continuous_patch_to_selected_source"
        )
    except (TypeError,ValueError,ArithmeticError,ZeroDivisionError,
            OverflowError):
        return False


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-bundle",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Research evidence must not be overwritten")
    bundle_digest=hashlib.sha256(args.paired_bundle.read_bytes()).hexdigest()
    with np.load(args.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected original paired NPZ layout")
        if (str(data["problem"].item())!="DTLZ5" or
            int(data["M"].item())!=4 or int(data["K"].item())!=1):
            raise ValueError("Wrong original MOP or dimensions")
        decisions=data["decisions"].copy()
    result=certify_dtlz5_patch_to_source_cover(decisions)
    if not replay_dtlz5_patch_to_source_cover(decisions,result):
        raise ArithmeticError("Independent full source proof replay failed")
    result["paired_artifact_sha256"]=bundle_digest
    with args.report.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    bound=_decode(result["continuous_patch_to_actual_source_l2_upper"])
    print("Actual selected original DTLZ5 patch cover upper:",float(bound))
    print("Paired PS/F SHA256:",bundle_digest)
    print("Whole GT certificate:",result["scientific_certification"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
