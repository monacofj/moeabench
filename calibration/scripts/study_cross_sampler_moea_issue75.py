#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Issue #75: compare Q_CLOSENESS under independent PF sampling *laws*.

Sources:
 (A) MoeaBench's existing deterministic Pareto-set/PF sampling, with
     reproducible unique-reservoir subsampling.
 (B) DTLZ1 uniform objective-simplex area / DTLZ2 uniform positive sphere area,
     independently constructed without the library PF sampler.
 (C) DPF5 exploratory randomized *decision*-space Pareto set evaluated via
     the MOP evaluator, explicitly NOT uniform objective-surface area.

Run real NSGA-II to obtain unchanged P0/Pmid/Pfinal *fronts*, use exact archived
GT, normalization and K-specific ECDF, and score every (source, sample seed)
hypothesis through the production qscore.q_closeness. Historical zero-residue
clinical Q is reproduced as a control for each final front. No certified files
are edited and no arbitrary Q drift cutoff is declared.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np
from scipy.spatial import cKDTree
import moeabench as mb

from moeabench.diagnostics import baselines
from calibration.scripts.independent_pf_sources_issue75 import normalized_independent_pf
from calibration.scripts.study_real_moea_q_sensitivity import (
    _analytical_samples, _moea_trial, _digest,
)
from calibration.scripts.study_ideal_residue_robustness import _stats, _region_hist


def summarize_law_shift(trials, samples):
    """Split cross-source drift from within-source seed noise.

    Clipped-only fronts cannot establish statistical stability and are
    separately counted rather than averaged away. Different laws may
    *legitimately* produce different epsilons because 'uniform on Pareto
    front' was under-specified by the original deterministic sampler.
    """
    summary={"by_front":[],"non_clipped_fronts":0,"clipped_all_zero_fronts":0,
             "max_cross_law_q_median_shift_nonclipped":None,
             "max_within_law_q_seed_span_nonclipped":None,
             "cross_law_shift_exceeds_within_law_seed_span":None,
             "max_cross_law_epsilon_median_shift":None,
             "not_certified":True}
    built=[float(v["eps_distance_median"]) for k,v in samples.items()
           if k.startswith("library_")]
    external=[float(v["eps_distance_median"]) for k,v in samples.items()
              if k.startswith("independent_")]
    if len(built)<2 or len(external)<2:
        raise ValueError("Both laws require replicated sampling seeds")
    delta=abs(float(np.median(built))-float(np.median(external)))
    summary["max_cross_law_epsilon_median_shift"]=delta
    shifts=[]
    within_spans=[]
    for trial in trials:
        for gen,front in trial["fronts"].items():
            byseed=front["q_by_analytical_seed"]
            left=[float(v) for k,v in byseed.items() if k.startswith("library_")]
            right=[float(v) for k,v in byseed.items() if k.startswith("independent_")]
            if len(left)<2 or len(right)<2:
                raise ValueError("Missing cross-law sample Q values")
            all_zero=all(x==0.0 for x in left+right)
            all_one=all(x==1.0 for x in left+right)
            informative=not (all_zero or all_one)
            difference=abs(float(np.median(left))-float(np.median(right)))
            within=float(max(max(left)-min(left),max(right)-min(right)))
            if informative:
                shifts.append(difference)
                within_spans.append(within)
            if all_zero: summary["clipped_all_zero_fronts"]+=1
            if informative: summary["non_clipped_fronts"]+=1
            summary["by_front"].append({
                "optimizer_seed":trial["seed"],"generation":gen,
                "raw_closeness_median":front["raw_closeness_median"],
                "front_sha256":front["front_sha256"],
                "old_q":front["archived_zero_residue_q"],
                "library_q":_stats(left),
                "independent_q":_stats(right),
                "median_q_shift":difference,
                "within_law_q_seed_span_max":within,
                "non_clipped":informative,
                "all_zero":all_zero,"all_one":all_one,
            })
    if shifts:
        summary["max_cross_law_q_median_shift_nonclipped"]=max(shifts)
        summary["max_within_law_q_seed_span_nonclipped"]=max(within_spans)
        # Descriptive scale comparison ONLY. This is not a hypothesis test
        # or a justified clinical-acceptance cutoff.
        summary["cross_law_shift_exceeds_within_law_seed_span"]=bool(
            max(shifts)>max(within_spans)
        )
    return summary


def analyze(name, *, moea_seeds=(1,7), ideal_seeds=(123,124,125,126),
            population=40, generations=60, n_fresh=1024):
    if name not in ("DTLZ1","DTLZ2","DPF5"):
        raise ValueError("No independent-source study protocol for "+name)
    if (population<12 or population%4 or generations<4 or n_fresh<2 or
            len(moea_seeds)<2 or len(set(moea_seeds))!=len(moea_seeds) or
            len(ideal_seeds)<2 or len(set(ideal_seeds))!=len(ideal_seeds)):
        raise ValueError("Require two distinct MOEA and two distinct PF seeds")
    reference=ROOT/"moeabench/diagnostics/resources/references"/name/"calibration_package.npz"
    with np.load(reference,allow_pickle=False) as package:
        GT=np.asarray(package["gt_norm"],dtype=float)
        original_ideal=np.asarray(package["ideal"],dtype=float)
        original_nadir=np.asarray(package["nadir"],dtype=float)
        import json as _json
        with (ROOT/"moeabench/diagnostics/resources/baselines_v0.14.0.json").open() as fd:
            baserec=_json.load(fd)["problems"][name]
        if not np.array_equal(GT,np.asarray(baserec["gt_reference"])):
            raise ValueError("Baseline GT and packaged GT do not match")
        internal,ideal,span=_analytical_samples(
            name,GT,package,n_fresh=n_fresh,seeds=ideal_seeds
        )
    if population!=baselines.snap_k(population):
        raise ValueError("Experimental K must match actual frozen ECDF resolution")
    s_k=float(baselines.get_resolution_factor_k(GT,population,seed=0))
    archive={int(k):np.asarray(d["closeness"]["rand_ecdf"],dtype=float)
             for k,d in baserec.items()
             if k.isdigit() and "closeness" in d}
    if population not in archive:
        raise ValueError("Frozen K-specific Closeness ECDF not available")
    native=getattr(mb.mops,name)()
    tree=cKDTree(GT)
    samples={}
    for seed,rec in internal.items():
        samples["library_"+seed]={
            "eps_distance_median":rec["eps_distance_median"],
            "sha256":rec["sha256"],"regions_f1":rec["regions_f1"],
            "source":"library_pf_unique_reservoir",
        }
    independent_metadata={}
    for seed in ideal_seeds:
        fresh,metadata=normalized_independent_pf(
            name,n_points=n_fresh,seed=int(seed),ideal=ideal,
            nadir=original_nadir,gt=GT,mop=native,
        )
        distances=tree.query(fresh,k=1)[0]
        key="independent_"+str(seed)
        samples[key]={
            "eps_distance_median":float(np.median(distances)),
            "sha256":_digest(fresh),"regions_f1":_region_hist(fresh,GT),
            "source":metadata["law"],
        }
        independent_metadata[str(seed)]=metadata
    raw_trials=[]
    for optimizer_seed in moea_seeds:
        # The existing real-front experiment already performs full
        # independent audit/GT normalization/clinical Q agreement checks.
        raw_trials.append(_moea_trial(
            name,GT,s_k,archive,seed=int(optimizer_seed),
            population=population,generations=generations,
            normalized_ideal=ideal,normalized_span=span,
            analytical_samples=samples,
        ))
    result=summarize_law_shift(raw_trials,samples)
    return {
        "status":"NONCERTIFIED_CROSS_SAMPLER_REAL_MOEA_Q_STUDY",
        "problem":name,"surface_law":"independent_uniform_area" if name!="DPF5"
                 else "exploratory_decision_space_density_unknown",
        "optimizer_seeds":list(moea_seeds),"analytical_seeds":list(ideal_seeds),
        "population":population,"generations":generations,"K":population,
        "s_K":s_k,"GT_sha256":_digest(GT),
        "GT_points":len(GT),"archived_ECDF_sha256":_digest(archive[population]),
        "samples":{k:{
            "source":v["source"],"eps_distance_median":v["eps_distance_median"],
            "ideal_res":v["eps_distance_median"]/s_k,
            "sha256":v["sha256"],"regions_f1":v["regions_f1"]
        } for k,v in samples.items()},
        "independent_source_provenance":independent_metadata,
        "fronts":raw_trials,
        "summary":result,
        "limitations":[
            "MOP PF distribution and externally specified objective-area law "
            "are different reference measures; difference does not by itself "
            "indicate a numerical implementation defect.",
            "DPF5 independent decision sampler is NOT area-uniform and may "
            "oversample some objective regions.",
            "Two real NSGA-II trajectories and four analytical seeds are a "
            "bounded sensitivity experiment, not a certified statistical tolerance.",
        ],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem",action="append",dest="problems",required=True)
    p.add_argument("--moea-seed",action="append",dest="moea_seeds",type=int)
    p.add_argument("--ideal-seed",action="append",dest="ideal_seeds",type=int)
    p.add_argument("--population",type=int,default=40)
    p.add_argument("--generations",type=int,default=60)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    output=a.output.resolve()
    blocked=(ROOT/"moeabench/diagnostics/resources").resolve()
    reports=(ROOT/"calibration/reports").resolve()
    if output.exists() or output.is_relative_to(blocked) or output.is_relative_to(reports):
        p.error("Refusing overwrite or certified artifact destination")
    results={"status":"NONCERTIFIED_CROSS_SAMPLER_REAL_MOEA_Q_STUDY",
             "results":{},"failures":{}}
    for name in a.problems:
        try:
            rec=analyze(name,moea_seeds=tuple(a.moea_seeds or (1,7)),
                        ideal_seeds=tuple(a.ideal_seeds or (123,124,125,126)),
                        population=a.population,generations=a.generations)
            results["results"][name]=rec
            print(json.dumps({
                "problem":name,"status":rec["status"],
                "K":rec["K"],
                "method":rec["surface_law"],
                "library_epsilon":_stats([
                    x["ideal_res"] for k,x in rec["samples"].items()
                    if k.startswith("library_")]),
                "external_epsilon":_stats([
                    x["ideal_res"] for k,x in rec["samples"].items()
                    if k.startswith("independent_")]),
                "max_cross_law_q_median_shift_nonclipped":
                  rec["summary"]["max_cross_law_q_median_shift_nonclipped"],
                "max_within_law_q_seed_span_nonclipped":
                  rec["summary"]["max_within_law_q_seed_span_nonclipped"],
                "cross_law_shift_exceeds_within_law_seed_span":
                  rec["summary"]["cross_law_shift_exceeds_within_law_seed_span"],
                "non_clipped_fronts":rec["summary"]["non_clipped_fronts"],
                "zero_clipped_fronts":rec["summary"]["clipped_all_zero_fronts"],
            }),flush=True)
        except (ValueError, TypeError, AssertionError, KeyError, OSError) as err:
            results["failures"][name]=f"{type(err).__name__}: {err}"
            print("FAILED",name,results["failures"][name],flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+"\n")
    if results["failures"]:
        raise SystemExit("Cross-sampler validation rejected one or more MOPs")


if __name__=="__main__":
    main()
