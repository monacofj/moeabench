# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Research-only original DTLZ6 M4/K1 full 448-source proof reach.

Independently classify exact binary decisions using the unit-radius theorem,
exact rational radial support when possible, and rigorous irrational-root
interval supporting strata. A negative or missing proof is never silently
promoted to a global PF certificate. A discrete candidate-pool covering
radius is purely observational; it does not bound the continuous front.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from moeabench.mops import DTLZ6
from moeabench.mops import _dtlz56_front as front

from .dtlz56_paired_symbolic_audit import _row_symbolic_result
from .dtlz56_dtlz6_radial_enclosure import (
    certify_dtlz6_radial_interval_pf, replay_dtlz6_radial_interval_pf,
)
from .dtlz56_paired_gt_evidence import _original_objectives


def classify_dtlz6_decision(row):
    x=tuple(float(v) for v in row)
    if len(x)!=4 or any(not np.isfinite(t) or t<0 or t>1 for t in x):
        raise ValueError("Original DTLZ6 M4 K1 feasible decision expected")
    basic=_row_symbolic_result(x,problem="DTLZ6",m=4,k=1)
    if basic["status"]=="symbolic_decision_global_pf":
        return ("proved_pf",basic["theorem"])
    if basic["status"]=="symbolic_decision_dominated":
        return ("dominated",basic["theorem"])
    proof=certify_dtlz6_radial_interval_pf(x[:3],x[3:],k=1)
    if proof.conclusion=="rigorously_global_pf_member":
        if not replay_dtlz6_radial_interval_pf(
            x[:3],x[3:],proof.to_dict(),k=1):
            raise AssertionError("Independent DTLZ6 irrational root proof replay failed")
        return ("proved_pf","rigorous_irrational_root_support")
    return ("unverified","insufficient_exact_theorem")


def _discrete_cover_distance(pool, selected):
    """Max over sampled source points of nearest selected distance; observational."""
    pool=np.asarray(pool,dtype=float)
    selected=np.asarray(selected,dtype=float)
    if not len(pool) or not len(selected):
        return None
    scales=np.maximum(np.ptp(pool,axis=0),1e-12)
    d=(pool[:,None,:]-selected[None,:,:])/scales
    return float(np.max(np.min(np.linalg.norm(d,axis=2),axis=1)))


def audit_source(*, n_points=16):
    if type(n_points) is not int or n_points!=16:
        raise ValueError("Bounded original source requires n=16")
    mop=DTLZ6(M=4,K=1)
    original_solver=front.scalarized_candidate
    original_reconstruct=front.decision_vector
    source=[]
    def observe_solver(weights,max_g,**kwargs):
        return original_solver(weights,max_g,**kwargs)
    def observe_reconstruct(candidate,**kwargs):
        decision=original_reconstruct(candidate,**kwargs)
        source.append(tuple(float(v) for v in decision))
        return decision
    try:
        front.scalarized_candidate=observe_solver
        front.decision_vector=observe_reconstruct
        selected=np.asarray(mop.ps(n_points=n_points),dtype=float)
    finally:
        front.scalarized_candidate=original_solver
        front.decision_vector=original_reconstruct
    if len(source)!=448 or selected.shape!=(16,4):
        raise AssertionError("Incomplete original DTLZ6 source or selected PS")
    pool=np.asarray(source,dtype=float)
    aligned_pool,_=_original_objectives(pool,problem="DTLZ6",m=4,k=1)
    aligned_selected,_=_original_objectives(
        selected,problem="DTLZ6",m=4,k=1)
    rows=[]
    for i,x in enumerate(pool):
        status,basis=classify_dtlz6_decision(x)
        rows.append({"index":i,"status":status,"basis":basis,
                     "family":("sobol" if i<256 else "gaussian" if i<384
                               else "gamma"),
                     "decision_hex":[float(v).hex() for v in x]})
    selected_rows=[]
    for i,x in enumerate(selected):
        status,basis=classify_dtlz6_decision(x)
        selected_rows.append({
            "index":i,"status":status,"basis":basis,
            "decision_hex":[float(v).hex() for v in x]})
    def counts(v):
        c=Counter(t["status"] for t in v)
        return {k:c.get(k,0) for k in
                ("proved_pf","dominated","unverified")}
    eligible=np.asarray([r["index"] for r in rows
                         if r["status"]=="proved_pf"],dtype=int)
    comparison={
        "status":"unverified_insufficient_proved_candidates",
        "scientific_certification":"unverified",
    }
    if len(eligible)>=12 and all(r["status"]=="proved_pf"
                                 for r in selected_rows[:4]):
        # Counterfactual only: preserve four source anchors, use same
        # normalized objective-spatial greedy metric as production.
        anchors=aligned_selected[:4]
        full_pf=np.vstack((anchors,aligned_pool))
        scales=np.maximum(np.ptp(full_pf,axis=0),1e-12)
        eligible_y=aligned_pool[eligible]
        nearest=np.min(np.sum(((eligible_y[:,None,:]-anchors[None,:,:])
                               /scales)**2,axis=2),axis=1)
        chosen=[]
        for _ in range(12):
            j=int(np.argmax(nearest))
            chosen.append(int(eligible[j]))
            distance=np.sum(((eligible_y-eligible_y[j])/scales)**2,axis=1)
            nearest=np.minimum(nearest,distance)
            nearest[j]=-1.
        alternative=np.vstack((anchors,aligned_pool[chosen]))
        # Empirical directed covering radius evaluated on the SAME
        # pool of independently proved source decisions for both selections.
        proof_y=aligned_pool[eligible]
        original_cover=_discrete_cover_distance(
            proof_y,aligned_selected)
        new_cover=_discrete_cover_distance(
            proof_y,alternative)
        comparison={
            "status":"observational_proved_only_counterfactual",
            "selected_proved_candidate_indices":chosen,
            "selected_count":16,
            "all_nonanchor_exact_pf_members":True,
            "original_proved_source_pool_cover_radius":original_cover,
            "counterfactual_proved_source_pool_cover_radius":new_cover,
            "positive_g_selected_count":int(sum(
                pool[i,3]>0 for i in chosen)),
            "finite_source_only_not_continuous_coverage":True,
            "scientific_certification":"unverified",
        }
    return {
        "schema":"dtlz6_original_m4k1_source_448_proof_reach_v1",
        "n_source_candidates":len(source),
        "n_selected":len(selected),
        "pool_counts":counts(rows),
        "selected_counts":counts(selected_rows),
        "by_family":{family:counts([
            r for r in rows if r["family"]==family])
            for family in ("sobol","gaussian","gamma")},
        "selected_rows":selected_rows,
        "candidate_rows":rows,
        "counterfactual":comparison,
        "continuous_front_coverage":"unverified",
        "indicator_fitness":"unverified",
        "scientific_certification":"unverified",
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Research reports cannot be overwritten")
    report=audit_source()
    with args.report.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:report[key] for key in
        ("pool_counts","selected_counts","by_family","counterfactual",
         "scientific_certification")},sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
