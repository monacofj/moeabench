# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded stratified diagnosis of the *source* DTLZ5/6 candidate-ray pool.

The original M>=4 PS sampler generates Sobol boundary, positive Gaussian,
and sparse Gamma preference weights before choosing a full reference by
greedy maximin selection. This opt-in experiment recreates that exact
source mechanism for M=4, K=1, nominal n_points=64 (pool=256), and
evaluates 8 deterministic, evenly spaced indices in EACH source stratum:
24 rays for EACH original problem. It calls the real production
scalarized_candidate/decision_vector helpers, with hard per-solve node
budget, and reports independent objective-image alignment and exact
decision-image proof reach.

No .ps(), final maximin selection, inferred population rate, or GT
certification is performed. A result is diagnostic of selected candidate
rays, NOT representative of the 64-point PS/F reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from moeabench.mops import DTLZ5, DTLZ6
from moeabench.mops._dtlz56_front import (
    scalarized_candidate, decision_vector,
)
from .dtlz56_small_scalarization_pilot import _git_commit, _version
from .dtlz56_paired_gt_evidence import _original_objectives
from .dtlz56_paired_symbolic_audit import audit_symbolic_candidate
from .dtlz56_dtlz6_radial_enclosure import (
    audit_dtlz6_irrational_radial_rows,
)
from .dtlz56_dtlz6_support_proximity import (
    certify_dtlz6_support_proximity,
)

SCHEMA = "moeabench_dtlz56_stratified_ray_diagnostic_v1"
NOMINAL_N = 64
SOURCE_POOL_SIZE = 256
INDICES = {
    "sobol_boundary": tuple(range(0,256,32)),
    "positive_gaussian": tuple(range(0,128,16)),
    "sparse_gamma": tuple(range(0,64,8)),
}


def _source_weights(problem, *, m=4, k=1):
    """Reproduce original PS candidate ray pipeline with same RNG ordering.

    This is a research mirror of _dtlz56_front.sample_original_dtlz56_ps,
    deliberately not a replacement public sampler. Any source change
    requires an explicit comparison/audit before new experiments.
    """
    if problem not in ("DTLZ5","DTLZ6") or type(m) is not int or m!=4:
        raise ValueError("This fixed experiment supports original M=4 only")
    if type(k) is not int or k!=1:
        raise ValueError("This fixed experiment supports K=1 only")
    mop = (DTLZ5 if problem=="DTLZ5" else DTLZ6)(M=m,K=k)
    size = SOURCE_POOL_SIZE
    sobol=qmc.Sobol(d=m,scramble=True,seed=42+m)
    draws=sobol.random_base2(8)[:size]
    x=np.zeros((size,m+k-1),dtype=float)
    x[:,0]=draws[:,0]
    x[:,1:m-1]=draws[:,2:m]
    face=np.arange(size)%(m-2)
    x[np.arange(size),1+face]=np.arange(size)%2
    gmax=k/4 if problem=="DTLZ5" else float(k)
    g_limit=np.where((np.arange(size)%4)==0,
                     gmax,np.minimum(gmax,1.5))
    g=g_limit*draws[:,1]**2
    if problem=="DTLZ5":
        x[:,m-1:]=(0.5+np.sqrt(g/k))[:,None]
    else:
        x[:,m-1:]=((g/k)**10)[:,None]
    boundary=np.maximum(mop.evaluation(x)["F"],1e-12)
    rng=np.random.default_rng(42+m)
    gaussian=np.abs(rng.standard_normal((size//2,m)))+1e-10
    gamma=rng.gamma(shape=0.35,scale=1.0,size=(size//4,m))
    parts={
        "sobol_boundary":boundary,
        "positive_gaussian":gaussian,
        "sparse_gamma":gamma,
    }
    selected=[]
    for stratum,indices in INDICES.items():
        array=parts[stratum]
        for index in indices:
            weights=array[index].copy()
            weights=np.maximum(weights,1.01e-7*float(weights.max()))
            selected.append((stratum,index,weights))
    return mop,selected


def run_stratified_diagnostic(*,problem,m=4,k=1,
                              abs_tol=1e-5,max_nodes=5000):
    if not np.isfinite(abs_tol) or not 0<abs_tol<=1e-4:
        raise ValueError("abs_tol must lie in (0,1e-4]")
    if type(max_nodes) is not int or not 1<=max_nodes<=12000:
        raise ValueError("max_nodes must be an integer in [1,12000]")
    mop, rays=_source_weights(problem,m=m,k=k)
    gmax=k/4 if problem=="DTLZ5" else float(k)
    decisions=[]
    optimizer=[]
    for stratum,index,w in rays:
        solved=scalarized_candidate(
            w,gmax,abs_tol=abs_tol,max_nodes=max_nodes,
        )
        x=decision_vector(solved,M=m,K=k,problem=problem)
        decisions.append(x)
        optimizer.append({
            "stratum":stratum,
            "source_pool_index":index,
            "normalized_weights":(w/np.max(w)).tolist(),
            "optimizer_g":float(solved.g),
            "scale":float(solved.scale),
            "abs_gap":float(solved.abs_gap),
            "lower_bound":float(solved.lower_bound),
            "nodes":int(solved.nodes),
        })
    x=np.asarray(decisions,dtype=float)
    f=mop.evaluation(x)["F"]
    independently, g=_original_objectives(x,problem=problem,m=m,k=k)
    residual=float(np.max(np.abs(independently-f)))
    per_row=audit_symbolic_candidate(
        x,f,problem=problem,m=m,k=k,
    )
    if per_row["independent_objective_alignment"]!="pass":
        raise AssertionError("Original source equations disagree with evaluator")

    irrational=None
    proximity=None
    if problem=="DTLZ6":
        irrational=audit_dtlz6_irrational_radial_rows(
            x,f,m=m,k=k,
        )
        proximity=[]
        for row in x:
            cert=certify_dtlz6_support_proximity(
                row[:m-1],row[m-1:],k=k,
            )
            proximity.append({
                "status":cert.conclusion,
                "distance_upper":(
                    cert.certified_l2_distance_upper
                    if cert.conclusion=="rigorously_within_certified_pf_distance"
                    else None
                ),
                "supporting_objective":cert.supporting_objective,
            })

    by_source={}
    for stratum in INDICES:
        members=[
            (index, record) for index,record in enumerate(optimizer)
            if record["stratum"]==stratum
        ]
        by_source[stratum]={
            "count":len(members),
            "global_pf_conditional_exact_g":
                sum(per_row["rows"][index]["status"]==
                    "symbolic_decision_global_pf" for index,_ in members),
            "dominated_conditional_exact_g":
                sum(per_row["rows"][index]["status"]==
                    "symbolic_decision_dominated" for index,_ in members),
            "global_pf_conditional_irrational_g":
                (None if irrational is None else sum(
                    irrational["rows"][index]["status"]==
                    "symbolic_decision_global_pf" for index,_ in members
                )),
            "proximity_to_proved_pf_bound_count":
                (None if proximity is None else sum(
                    proximity[index]["status"]==
                    "rigorously_within_certified_pf_distance"
                    for index,_ in members
                )),
            "max_abs_gap":max(record["abs_gap"] for _,record in members),
            "max_nodes":max(record["nodes"] for _,record in members),
        }

    return {
        "schema":SCHEMA,
        "problem":problem,"M":m,"K":k,
        "source_sha":_git_commit(),
        "numpy":_version("numpy"),"scipy":_version("scipy"),
        "method":"source_mirrored_preselection_stratified_scalarization",
        "nominal_full_sampler_points":NOMINAL_N,
        "full_source_pool_size":SOURCE_POOL_SIZE,
        "stratified_ray_count":len(rays),
        "indices":{key:list(v) for key,v in INDICES.items()},
        "abs_tol":float(abs_tol),"max_nodes":max_nodes,
        "decisions_sha256":hashlib.sha256(x.tobytes()).hexdigest(),
        "objectives_sha256":hashlib.sha256(f.tobytes()).hexdigest(),
        "independent_max_abs_error":residual,
        "decision_rows":x.tolist(),
        "objective_rows":f.tolist(),
        "original_evaluated_g":g.tolist(),
        "optimizer":optimizer,
        "exact_rational_g_audit":per_row,
        "irrational_g_global_support":irrational,
        "certified_pf_proximity":proximity,
        "stratum_diagnostics":by_source,
        "full_gt_pareto_membership":"unverified",
        "full_gt_coverage":"unverified",
        "scientific_certification":"unverified",
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem",choices=("DTLZ5","DTLZ6"),required=True)
    parser.add_argument("--m",type=int,default=4)
    parser.add_argument("--k",type=int,default=1)
    parser.add_argument("--abs-tol",type=float,default=1e-5)
    parser.add_argument("--max-nodes",type=int,default=5000)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    report=run_stratified_diagnostic(
        problem=args.problem,m=args.m,k=args.k,
        abs_tol=args.abs_tol,max_nodes=args.max_nodes,
    )
    payload=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x",encoding="utf-8") as handle:
            handle.write(payload)
    else:
        print(payload,end="")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
