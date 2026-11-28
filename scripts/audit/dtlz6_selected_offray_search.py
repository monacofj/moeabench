# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, deterministic numerical witness discovery + exact DTLZ6 proof.

The original M4/K1 n16 PS/F is regenerated unchanged. SLSQP is a
DISCOVERY heuristic only: numeric objectives, optimizer success flags,
and its approximate margin can never establish Pareto dominance.
Each proposed original feasible IEEE-754 decision is then fed to the
INDEPENDENT exact rational DTLZ6 interval strict-dominance verifier.
The report says either CERTIFIED exact dominator or UNVERIFIED.
No positive membership or front/indicator coverage is inferred.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .dtlz56_exact_same_ray import _encode
from .dtlz6_exact_offray_witness import (
    certify_dtlz6_offray_dominator,replay_dtlz6_offray_dominator,
)

SCHEMA="DTLZ6-M4K1-ORIGINAL-SELECTED-BOUNDED-OFFRAY-SEARCH-V1"
SELECTED_ROWS=(6,7,10,11,12)


def _objective_from_radial(z):
    x1,x2,x3,g=(float(v) for v in z)
    r=1+g
    a=np.pi*x1/2
    b=np.pi*(1+2*g*x2)/(4*r)
    c=np.pi*(1+2*g*x3)/(4*r)
    return r*np.array((
        np.cos(a)*np.cos(b)*np.cos(c),
        np.cos(a)*np.cos(b)*np.sin(c),
        np.cos(a)*np.sin(b),
        np.sin(a),
    ),dtype=float)


def _trial_seeds(source):
    x1,x2,x3,tail=(float(v) for v in source)
    g=tail**0.1
    seeds=[(x1,x2,x3,g)]
    for dg in (-.25,-.10,.10,.25):
        gp=min(1.,max(0.,g+dg))
        for dx in (-.05,.05):
            seeds.append((
                min(1.,max(0.,x1+dx)),x2,x3,gp
            ))
    for x2p,x3p in ((0.,0.),(0.,1.),(1.,0.),(1.,1.)):
        seeds.append((x1,x2p,x3p,g))
    return seeds


def search_selected_offray(decisions,*,indices=SELECTED_ROWS,
                            maxiter=180):
    try:
        ps=np.asarray(decisions,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("Expected original source PS") from exc
    if ps.ndim!=2 or ps.shape[1]!=4 or not np.all(np.isfinite(ps)) or (
        np.any(ps<0) or np.any(ps>1)
    ):
        raise ValueError("Expected finite feasible original M4/K1 source")
    if type(maxiter) is not int or not 10<=maxiter<=1000:
        raise ValueError("Bounded iterations required")
    if len(indices)!=len(set(indices)) or any(
        type(i) is not int or i<0 or i>=len(ps) for i in indices
    ):
        raise ValueError("Selected indices must be unique in source")
    result=[]
    for i in indices:
        source=ps[i]
        source_g=source[3]**.1
        source_objectives=_objective_from_radial((
            source[0],source[1],source[2],source_g))
        best=None
        tested=0
        exact_certificate=None
        exact_witness=None
        for seed in _trial_seeds(source):
            def goal(v):
                return float(v[4])
            def constraint(v):
                return source_objectives-_objective_from_radial(v[:4])+v[4]
            start=np.asarray((*seed,1e-5),dtype=float)
            trial=minimize(goal,start,method="SLSQP",
                bounds=((0,1),(0,1),(0,1),(0,1),(-2,2)),
                constraints=({"type":"ineq","fun":constraint},),
                options={"maxiter":maxiter,"ftol":1e-12,"disp":False})
            proposals=(trial.x[:4],)
            for proposal in proposals:
                decision=(
                    float(proposal[0]),float(proposal[1]),
                    float(proposal[2]),float(proposal[3])**10,
                )
                if any(not np.isfinite(v) or not 0<=v<=1 for v in decision):
                    continue
                observed=float(np.max(
                    _objective_from_radial(proposal)-source_objectives))
                tested+=1
                if best is None or observed<best:
                    best=observed
                if observed<0:
                    proof=certify_dtlz6_offray_dominator(source,decision)
                    if proof["conclusion"]=="rigorously_strictly_dominated":
                        if not replay_dtlz6_offray_dominator(
                            source,decision,proof):
                            raise ArithmeticError("Independent exact proof failed")
                        exact_certificate=proof
                        exact_witness=[float(v).hex() for v in decision]
                        break
            if exact_certificate is not None:
                break
        result.append({
            "index":i,
            "original_decision_hex":[float(v).hex() for v in source],
            "numerical_proposals_evaluated":tested,
            "best_numeric_max_coordinate_difference":best,
            "conclusion":(
                "rigorously_strictly_dominated" if exact_certificate is not None
                else "unverified"
            ),
            "certified_witness_hex":exact_witness,
            "exact_rational_proof":exact_certificate,
        })
    return {
        "schema":SCHEMA,"problem":"DTLZ6","M":4,"K":1,
        "original_selected_n":len(ps),
        "selected_decision_sha256":hashlib.sha256(
            b"".join(float(v).hex().encode("ascii")+b";"
                     for row in ps for v in row)).hexdigest(),
        "bounded_iterations_per_start":maxiter,
        "target_indices":list(indices),
        "rigorously_dominated_indices":[r["index"] for r in result
            if r["conclusion"]=="rigorously_strictly_dominated"],
        "still_unverified_indices":[r["index"] for r in result
            if r["conclusion"]=="unverified"],
        "rows":result,
        "numerical_search_only":"never_a_dominance_or_pf_oracle",
        "unproved_rows_do_not_imply_global_pf":"unverified",
        "full_front_or_indicator_quality":"unverified",
        "scientific_certification":"unverified",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-bundle",required=True,type=Path)
    p.add_argument("--report",required=True,type=Path)
    args=p.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("No overwrite of research evidence")
    digest=hashlib.sha256(args.paired_bundle.read_bytes()).hexdigest()
    with np.load(args.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected original PS/F members")
        if (str(data["problem"].item())!="DTLZ6" or
            int(data["M"].item())!=4 or int(data["K"].item())!=1
        ):
            raise ValueError("Expected original DTLZ6 M4/K1")
        ps=data["decisions"].copy()
    result=search_selected_offray(ps)
    if result["original_selected_n"]!=16:
        raise ValueError("Original selected sample must have 16 rows")
    # Crucial independent exact replay of every proposed accepted witness.
    for r in result["rows"]:
        if r["conclusion"]!="rigorously_strictly_dominated":
            continue
        source=ps[r["index"]]
        witness=[float.fromhex(v) for v in r["certified_witness_hex"]]
        if not replay_dtlz6_offray_dominator(
            source,witness,r["exact_rational_proof"]):
            raise ArithmeticError("Archived strict witness failed exact replay")
    result["paired_artifact_sha256"]=digest
    with args.report.open("x",encoding="utf-8") as fp:
        json.dump(result,fp,sort_keys=True,indent=2,allow_nan=False)
        fp.write("\n")
    print("DTLZ6 rigorous positive/zero g offray dominated selected indices:",
          result["rigorously_dominated_indices"])
    print("DTLZ6 offray search still inconclusive indices:",
          result["still_unverified_indices"])
    for row in result["rows"]:
        print("DTLZ6 selected index",row["index"],"result",row["conclusion"],
              "best numeric violation (NOT proof)",
              row["best_numeric_max_coordinate_difference"])
    print("DTLZ6 scientific certification:",result["scientific_certification"])


if __name__=="__main__":
    main()
