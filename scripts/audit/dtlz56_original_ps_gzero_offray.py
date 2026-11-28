# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact offray g0 continuous PF curve screen on actual selected PS/F.

Original DTLZ5/6 M4 K1 selected binary decisions, unchanged. Every
approved decision is strictly dominated by a feasible mathematical
point on the COMPLETE original g=0 Pareto curve, with a final
objective exactly matching the candidate's last objective.
All other source rows ABSTAIN; this is not whole-front certification.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np

from .dtlz56_gzero_offray_dominance import (
    certify_gzero_curve_offray_dominance,
    replay_gzero_curve_offray_dominance,
)
from .dtlz56_exact_same_ray import _encode

SCHEMA="DTLZ56-M4K1-ORIGINAL-SELECTED-PS-GZERO-OFFRAY-LEDGER-V1"


def certify_original_ps_gzero_offray(decisions,*,problem):
    if problem not in ("DTLZ5","DTLZ6"):
        raise ValueError("Expected original DTLZ5/6")
    try:
        ps=tuple(tuple(Fraction.from_float(float(v)) for v in row)
                 for row in decisions)
    except (TypeError,ValueError,OverflowError) as e:
        raise ValueError("Expected finite binary original PS") from e
    if not ps or any(len(row)!=4 or any(not 0<=v<=1 for v in row)
                     for row in ps):
        raise ValueError("Original M4/K1 PS must be nonempty and feasible")
    result=[]
    for i,row in enumerate(ps):
        proof=certify_gzero_curve_offray_dominance(row,problem=problem)
        if proof["conclusion"].startswith("rigorously_") and (
            not replay_gzero_curve_offray_dominance(
                row,proof,problem=problem)):
            raise ArithmeticError("Independent gzero offray proof replay failed")
        result.append({
            "index":i,
            "original_decision":[_encode(v) for v in row],
            "certificate":proof,
        })
    dominated=[
        x["index"] for x in result if
        x["certificate"]["conclusion"]==
        "rigorously_strictly_dominated_by_gzero_curve"
    ]
    return {
        "schema":SCHEMA,"problem":problem,"M":4,"K":1,
        "selected_n":len(ps),
        "selected_decision_sha256":hashlib.sha256(
            b"".join(float(v).hex().encode("ascii")+b";"
                     for row in ps for v in row)).hexdigest(),
        "rigorously_gzero_offray_dominated_indices":dominated,
        "rows":result,
        "all_unexcluded_membership":"unverified",
        "whole_gt_membership":(
            "rejected_has_dominated_member" if dominated else "unverified"),
        "scientific_certification":"unverified",
    }


def replay_original_ps_gzero_offray(decisions,record,*,problem):
    try:
        return record==certify_original_ps_gzero_offray(
            decisions,problem=problem)
    except (TypeError,ValueError,KeyError,ArithmeticError,
            OverflowError,ZeroDivisionError):
        return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-bundle",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args(argv)
    if a.report.exists():
        raise FileExistsError("Write-once research evidence required")
    digest=hashlib.sha256(a.paired_bundle.read_bytes()).hexdigest()
    with np.load(a.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected original PS/F artifact schema")
        problem=str(data["problem"].item())
        if problem not in ("DTLZ5","DTLZ6") or (
            int(data["M"].item())!=4 or int(data["K"].item())!=1
        ):
            raise ValueError("Wrong original problem or dimension")
        ps=data["decisions"].copy()
    report=certify_original_ps_gzero_offray(ps,problem=problem)
    if not replay_original_ps_gzero_offray(ps,report,problem=problem):
        raise ArithmeticError("Independent source ledger failed replay")
    report["paired_artifact_sha256"]=digest
    with a.report.open("x",encoding="utf-8") as fp:
        json.dump(report,fp,indent=2,sort_keys=True,allow_nan=False)
        fp.write("\n")
    print(problem,"strict continuous gzero OFFRAY dominated source indices:",
          report["rigorously_gzero_offray_dominated_indices"])
    print(problem,"whole GT status:",report["whole_gt_membership"])


if __name__=="__main__":
    main()
