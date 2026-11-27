# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Original source PS exact open-angular exclusion ledger, DTLZ5/6 M4K1.

Use EXACT IEEE-754 decision bits, not stored numeric objectives or
an assumed approximately zero radial distance. In original DTLZ5 K1,
g=(tail-1/2)^2 is positive precisely when tail != 1/2.
In original DTLZ6 K1, g=tail**(1/10) is positive precisely when
tail>0, even when irrational. The universal exact-real
same-ray exclusion theorem then proves every selected decision
with both 0<x2<1 AND 0<x3<1 strictly dominated.

For DTLZ6 no rational tenth-root is substituted into a purported
exact proof. The original real analytic witness is:
 q=max_i |g*(2xi-1)/(1+g)|<g/(1+g);
 gmin=q/(1-q); gp=(g+gmin)/2; xi'=1/2+
 d_i*(1+gp)/(2gp); tail'=gp**10. This is feasible and
 scales every objective by (1+gp)/(1+g) <1.
For DTLZ5 the tail'=1/2+sqrt(gp) works since gp<=1/4.

This is an independent necessary-condition screening ledger.
Boundary rows abstain, and neither finite F exactness nor whole GT
coverage/indicator quality is accepted.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np

from .dtlz56_exact_same_ray import _encode
from .dtlz56_open_inner_angle_exclusion import (
    certify_open_inner_angle_exclusion,
    replay_open_inner_angle_exclusion,
)

SCHEMA="dtlz56_m4k1_original_ps_open_inner_exclusion_v1"


def certify_original_ps_open_inner_exclusion(decisions, *,problem):
    if problem not in ("DTLZ5","DTLZ6"):
        raise ValueError("Expected original DTLZ5 or DTLZ6")
    try:
        ps=tuple(tuple(Fraction.from_float(float(v)) for v in row)
                 for row in decisions)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("Nonfinite or unsupported original PS") from exc
    if not ps or any(len(r)!=4 or any(not 0<=v<=1 for v in r)
                     for r in ps):
        raise ValueError("Expected feasible original M4 K1 PS")
    results=[]
    for i,row in enumerate(ps):
        x1,x2,x3,tail=row
        is_positive=(tail!=Fraction(1,2)
                     if problem=="DTLZ5" else tail>0)
        open_angular=0<x2<1 and 0<x3<1
        dominated=is_positive and open_angular
        rational_witness=None
        if dominated and problem=="DTLZ5":
            g=(tail-Fraction(1,2))**2
            exact=certify_open_inner_angle_exclusion(
                row[:3],g=g,problem=problem,k=1)
            if not replay_open_inner_angle_exclusion(
                row[:3],exact,g=g,problem=problem,k=1):
                raise ArithmeticError("Exact rational row witness did not replay")
            rational_witness=exact
        results.append({
            "index":i,
            "decision":[_encode(v) for v in row],
            "original_radial_g_strictly_positive":is_positive,
            "all_inner_angular_coordinates_open":open_angular,
            "conclusion":(
                "rigorously_strictly_dominated"
                if dominated else "unverified"
            ),
            "exact_rational_witness":rational_witness,
            "symbolic_irrational_g_witness":(
                "real_same_ray_gprime=(g+q/(1-q))/2;tailprime=gprime^10"
                if dominated and problem=="DTLZ6" else None),
        })
    bad=[x["index"] for x in results
         if x["conclusion"]=="rigorously_strictly_dominated"]
    digest=hashlib.sha256(
        b"".join(float(v).hex().encode("ascii")+b";"
                 for row in ps for v in row)
    ).hexdigest()
    return {
        "schema":SCHEMA,"problem":problem,"M":4,"K":1,
        "selected_count":len(ps),
        "selected_decisions_sha256":digest,
        "rigorously_dominated_open_inner_indices":bad,
        "rigorously_dominated_open_inner_count":len(bad),
        "rows":results,
        "necessary_pf_boundary_condition":"x2 or x3 exactly in {0,1} when g>0",
        "remaining_boundary_rows":"unverified",
        "original_F_exact_alignment":"unverified",
        "whole_reference_membership":(
            "rejected_has_dominated_member" if bad else "unverified"),
        "scientific_certification":"unverified",
    }


def replay_original_ps_open_inner_exclusion(decisions,report,*,problem):
    try:
        expected=certify_original_ps_open_inner_exclusion(
            decisions,problem=problem)
        return report==expected
    except (TypeError,ValueError,ArithmeticError,OverflowError,
            ZeroDivisionError,AssertionError):
        return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-bundle",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    args=p.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Do not overwrite scientific evidence")
    sha=hashlib.sha256(args.paired_bundle.read_bytes()).hexdigest()
    with np.load(args.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected source bundle members")
        problem=str(data["problem"].item())
        if problem not in ("DTLZ5","DTLZ6") or (
            int(data["M"].item())!=4 or int(data["K"].item())!=1
        ):
            raise ValueError("Wrong original source problem or dimensions")
        ps=data["decisions"].copy()
    report=certify_original_ps_open_inner_exclusion(ps,problem=problem)
    if not replay_original_ps_open_inner_exclusion(ps,report,problem=problem):
        raise ArithmeticError("Exact open-inner source ledger failed replay")
    report["paired_artifact_sha256"]=sha
    with args.report.open("x",encoding="utf-8") as out:
        json.dump(report,out,sort_keys=True,indent=2,allow_nan=False)
        out.write("\n")
    print(problem,"EXACT excluded interior source rows",
          report["rigorously_dominated_open_inner_indices"])
    print(problem,"remaining boundary member status",
          report["remaining_boundary_rows"])
    print(problem,"whole GT status",report["whole_reference_membership"])


if __name__=="__main__":
    main()
