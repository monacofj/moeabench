# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Source-bound exact last-objective cap exclusion for original DTLZ5/6.

Original M4 K1 selected decision bits are preserved; original DTLZ5
radial g is exact rational; original DTLZ6 radial g may be irrational
but an integer tenth-root enclosure supplies its rigorous lower bound.
Every strictly proved dominated source row is rejected, all other rows
ABSTAIN; never infer Pareto membership from a nonexclusion.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from .dtlz56_dtlz6_radial_enclosure import rational_tenth_root_enclosure
from .dtlz56_last_objective_cap_exclusion import (
    certify_last_objective_cap_exclusion,
    replay_last_objective_cap_exclusion,
)
from .dtlz56_exact_same_ray import _encode

SCHEMA="DTLZ56-M4K1-ACTUAL-PS-LAST-OBJECTIVE-CAP-V1"


def certify_original_ps_last_objective_cap(
    decisions,*,problem,root_bits=56,
):
    if problem not in ("DTLZ5","DTLZ6"):
        raise ValueError("Expected original DTLZ5/6")
    if type(root_bits) is not int or not 8<=root_bits<=80:
        raise ValueError("Unsupported integer tenth-root precision")
    try:
        ps=tuple(tuple(Fraction.from_float(float(x)) for x in row)
                 for row in decisions)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("Expected finite IEEE-754 original PS") from exc
    if not ps or any(len(row)!=4 or any(not 0<=x<=1 for x in row)
                     for row in ps):
        raise ValueError("Expected feasible original M4 K1 PS")
    out=[]
    for i,row in enumerate(ps):
        tail=row[3]
        if problem=="DTLZ5":
            g_lower=(tail-Fraction(1,2))**2
        else:
            g_lower,_=rational_tenth_root_enclosure(
                tail,bits=root_bits)
        proof=certify_last_objective_cap_exclusion(
            row[0],g_lower=g_lower,problem=problem,k=1)
        if proof["conclusion"].startswith("rigorously_") and (
            not replay_last_objective_cap_exclusion(
                row[0],proof,g_lower=g_lower,problem=problem,k=1)):
            raise ArithmeticError("Original source strict proof did not replay")
        out.append({
            "source_index":i,
            "decision":[_encode(v) for v in row],
            "certified_radial_g_lower":_encode(g_lower),
            "last_objective_proof":proof,
        })
    excluded=[x["source_index"] for x in out
              if x["last_objective_proof"]["conclusion"].startswith("rigorously_")]
    return {
        "schema":SCHEMA,"problem":problem,"M":4,"K":1,
        "selected_decision_count":len(ps),
        "selected_decisions_sha256":hashlib.sha256(
            b"".join(float(v).hex().encode("ascii")+b";"
                     for row in ps for v in row)
        ).hexdigest(),
        "root_bits":root_bits,
        "provably_dominated_last_objective_rows":excluded,
        "rows":out,
        "boundary_or_remaining_membership":"unverified",
        "whole_gt_membership":(
            "rejected_has_dominated_member" if excluded else "unverified"),
        "indicator_fitness":"unverified",
        "scientific_certification":"unverified",
    }


def replay_original_ps_last_objective_cap(
    decisions,record,*,problem,root_bits=56,
):
    try:
        return record==certify_original_ps_last_objective_cap(
            decisions,problem=problem,root_bits=root_bits)
    except (ValueError,TypeError,ArithmeticError,KeyError,
            OverflowError,ZeroDivisionError):
        return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-bundle",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    args=p.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Scientific evidence is write once")
    digest=hashlib.sha256(args.paired_bundle.read_bytes()).hexdigest()
    with np.load(args.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected original paired bundle fields")
        problem=str(data["problem"].item())
        if problem not in ("DTLZ5","DTLZ6") or (
            int(data["M"].item())!=4 or int(data["K"].item())!=1
        ):
            raise ValueError("Wrong problem/dimensions")
        ps=data["decisions"].copy()
    result=certify_original_ps_last_objective_cap(ps,problem=problem)
    if not replay_original_ps_last_objective_cap(
        ps,result,problem=problem):
        raise ArithmeticError("Independent original source proof replay failed")
    result["paired_artifact_sha256"]=digest
    with args.report.open("x",encoding="utf-8") as out:
        json.dump(result,out,sort_keys=True,indent=2,allow_nan=False)
        out.write("\n")
    print(problem,"last-objective rigorously dominated rows:",
          result["provably_dominated_last_objective_rows"])
    print(problem,"all other source rows:",result["boundary_or_remaining_membership"])
    print(problem,"whole GT status:",result["whole_gt_membership"])


if __name__=="__main__":
    main()
