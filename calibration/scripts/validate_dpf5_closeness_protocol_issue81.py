#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only DPF5 M3/D2 Closeness reference-law stability study (issue #81).

This validates only the EXPLICIT proposed source laws in ADR 0055:
  Ideal: independently sampled iid uniform Pareto-set parameters,
         retaining x1>1/3's 2/3 mass; no objective deduplication.
  Unfavorable: one corrected outward-only GT-normal half-blur per
         distinct archived GT row, each with equal weight.
  Intensity: original K-dependent median distance / s_K target 2,
         six-iteration controller with 5% fit tolerance.

Vary ideal seeds and sample sizes separately from sigma-fit seeds and
independent final negative-reference draws. Report empirical ranges;
do not invent a statistical acceptance threshold, scientific consensus,
or an official baseline. All canonical files remain untouched.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np
from scipy.spatial import cKDTree

from moeabench.mops import DPF5
from moeabench.diagnostics.baselines import get_resolution_factor_k
from calibration.scripts.generate_baselines import (
    calibrate_sigma, estimate_gt_normals, generate_half_normal_blur,
    N_SAMPLES
)

RESOURCE_DIR=(ROOT/"moeabench/diagnostics/resources").resolve()
FROZEN=RESOURCE_DIR/"baselines_v0.14.0.json"
PACKAGE=RESOURCE_DIR/"references/DPF5/calibration_package.npz"
REPORTS=(ROOT/"calibration/reports").resolve()
OFFICIAL=(ROOT/"calibration/audit_report.html").resolve()

def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(array,dtype=np.float64)).tobytes()).hexdigest()

def frozen_reference():
    source=json.loads(FROZEN.read_text())
    gt=np.asarray(source["problems"]["DPF5"]["gt_reference"],dtype=np.float64)
    with np.load(PACKAGE,allow_pickle=False) as original:
        npz=np.asarray(original["gt_norm"],dtype=np.float64)
        ideal=np.asarray(original["ideal"],dtype=np.float64)
        nadir=np.asarray(original["nadir"],dtype=np.float64)
    if (not np.array_equal(npz,gt) or gt.shape!=(3367,3) or
            not np.all(nadir>ideal)):
        raise ValueError("Frozen registered DPF5 GT and NPZ do not agree")
    model=DPF5(D=2,M=3)
    X=np.asarray(model.ps(n_points=10000),dtype=np.float64)
    F=np.asarray(model.evaluation(X)["F"],dtype=np.float64)
    reconstructed=np.unique((F-ideal)/(nadir-ideal),axis=0)
    distinct=np.unique(gt,axis=0)
    if (len(X)!=10000 or len(reconstructed)!=3367 or
            not np.allclose(reconstructed,distinct,atol=1e-12,rtol=0)):
        raise ValueError("Historical GT source support reconstruction failed")
    raw_D=int(np.count_nonzero(X[:,0] > 1/3))
    if raw_D!=6700:
        raise AssertionError("Original DPF5 Pareto-set branch semantics changed")
    return source,gt,ideal,nadir,model

def sample_independent_ideal(model,*,size,seed):
    if not isinstance(size,int) or size<64 or size>100000:
        raise ValueError("Expected independent size between 64 and 100000")
    if not isinstance(seed,int) or seed<0:
        raise ValueError("Expected nonnegative independent seed")
    if model.M!=3 or model.D!=2:
        raise ValueError("Only DPF5 with M=3,D=2 supported")
    random=np.random.default_rng(seed)
    params=random.random((size,2))
    X=np.zeros((size,model.N),dtype=np.float64)
    X[:,:2]=params
    X[:,2]=params[:,0]
    X[:,3:]=0.5
    F=np.asarray(model.evaluation(X)["F"],dtype=np.float64)
    if (F.shape!=(size,3) or not np.all(np.isfinite(F))
            or np.min(F)<-1e-12 or
            np.max(np.abs(np.linalg.norm(F,axis=1)-1))>1e-12):
        raise AssertionError("Analytical DPF5 sample not on nonnegative unit sphere")
    return F,params[:,0]>1/3

def _summary(values):
    v=np.asarray(values,dtype=np.float64)
    if v.ndim!=1 or not len(v) or not np.all(np.isfinite(v)):
        raise ValueError("Expected finite nonempty one-dimensional results")
    return {
        "n":len(v),
        "min":float(np.min(v)),
        "median":float(np.median(v)),
        "max":float(np.max(v)),
        "max_minus_min":float(np.max(v)-np.min(v)),
    }

def study(*,ks=(40,100,200),ideal_sizes=(4096,16384),
          ideal_seeds=(123,124,125,126),
          fit_seeds=(1100,1101,1102,1103),
          blur_seeds=(1600,1601,1602,1603)):
    ks=tuple(ks)
    ideal_sizes=tuple(ideal_sizes)
    ideal_seeds=tuple(ideal_seeds)
    fit_seeds=tuple(fit_seeds)
    blur_seeds=tuple(blur_seeds)
    for label,seq in [
        ("K",ks),("independent size",ideal_sizes),
        ("ideal seeds",ideal_seeds),("fit seeds",fit_seeds),
        ("blur seeds",blur_seeds)
    ]:
        if not seq or len(set(seq))!=len(seq) or not all(
                isinstance(x,int) and x>=0 for x in seq):
            raise ValueError(label+" must contain distinct nonnegative integers")
    if (any(k<2 or k>200 for k in ks)
            or any(size<64 or size>100000 for size in ideal_sizes)
            or len(set(fit_seeds).intersection(blur_seeds))):
        raise ValueError("Invalid K, sample sizes or overlapping reference seeds")

    original,gt,ideal,nadir,mop=frozen_reference()
    gt_tree=cKDTree(gt)
    normals=estimate_gt_normals(gt)
    if not (np.all(normals>=0)
            and np.allclose(np.linalg.norm(normals,axis=1),1,atol=1e-12)):
        raise AssertionError("Corrected main-branch outward PCA normals not loaded")

    # Calculate each ideal trajectory once, independently of K.  K only
    # changes the registered finite-GT s_K, not the geometric sample.
    ideals={}
    for size in ideal_sizes:
        row=[]
        for seed in ideal_seeds:
            physical,D=sample_independent_ideal(mop,size=size,seed=seed)
            obj=(physical-ideal)/(nadir-ideal)
            dist=gt_tree.query(obj,k=1)[0]
            if not (np.all(np.isfinite(dist)) and np.min(dist)>1e-10):
                raise AssertionError("Ideal sample overlaps the historical finite GT")
            row.append({
                "sample_size":size,"seed":seed,
                "true_D_fraction":float(np.mean(D)),
                "objective_sha256":_sha(obj),
                "median_geometric_distance":float(np.median(dist))
            })
        ideals[str(size)]=row

    by_K={}
    for K in ks:
        if str(K) not in original["problems"]["DPF5"]:
            raise ValueError("K is not registered in frozen DPF5 baseline")
        scale=float(get_resolution_factor_k(gt,K,seed=0))
        if not np.isfinite(scale) or scale<=0:
            raise AssertionError("Registered K-resolution invalid")
        # A SINGLE chosen ideal epsilon is used to compare negative ECDF
        # realizations: here the median of the largest ideal-sample seeds,
        # used for sensitivity only, not as a declared release estimate.
        eps_anchors=[x["median_geometric_distance"]/scale
                     for x in ideals[str(max(ideal_sizes))]]
        illustrative_epsilon=float(np.median(eps_anchors))
        per_fit=[]
        sigma_values=[]
        controller_values=[]
        final_medians=[]
        quantiles10,quantiles50,quantiles90=[],[],[]
        for fit_seed in fit_seeds:
            sigma,controller_median,controller_p95=calibrate_sigma(
                gt,normals,scale,fit_seed,tree=gt_tree)
            if (not np.isfinite(sigma) or sigma<=0 or
                    abs(controller_median-2.)>=.1):
                raise AssertionError("Existing six-iteration sigma controller did not meet 5% fit target")
            sigma_values.append(float(sigma))
            controller_values.append(float(controller_median))
            final=[]
            for negative_seed in blur_seeds:
                new=generate_half_normal_blur(
                    gt,normals,sigma,np.random.RandomState(negative_seed))
                if not (np.all(new>=gt) and np.all(np.any(new>gt,axis=1))):
                    raise AssertionError("One-sided blur no longer strictly dominated")
                raw=gt_tree.query(new,k=1)[0]/scale
                corrected=np.maximum(0.,raw-illustrative_epsilon)
                # Exactly the frozen schema's 200 sorted order statistics,
                # NOT a new interpolation convention.
                ordered=np.sort(corrected)
                positions=np.linspace(0,len(ordered)-1,N_SAMPLES).astype(int)
                ecdf=ordered[positions]
                med=float(np.median(raw))
                final_medians.append(med)
                q10=float(np.quantile(ecdf,.1))
                q50=float(np.quantile(ecdf,.5))
                q90=float(np.quantile(ecdf,.9))
                quantiles10.append(q10);quantiles50.append(q50);quantiles90.append(q90)
                final.append({
                    "blur_seed":negative_seed,
                    "blurred_objective_sha256":_sha(new),
                    "realized_raw_median_u":med,
                    "within_controller_5percent_band":bool(abs(med-2.)<.1),
                    "candidate_ecdf_sha256":_sha(ecdf),
                    "candidate_ecdf_p10_p50_p90":[q10,q50,q90],
                    "strict_parent_dominance_fraction":1.,
                })
            per_fit.append({
                "fit_seed":fit_seed,"sigma":float(sigma),
                "controller_median_u":float(controller_median),
                "controller_p95_u":float(controller_p95),
                "independent_null_draws":final,
            })
        by_K[str(K)]={
            "s_K":scale,
            "archived_sigma":float(original["problems"]["DPF5"][str(K)]["closeness"]["sigma"]),
            "ideal_epsilon_by_size":{
                str(size):{
                    "samples":[{
                        "seed":record["seed"],
                        "epsilon":record["median_geometric_distance"]/scale,
                        "true_D_fraction":record["true_D_fraction"],
                    } for record in ideals[str(size)]],
                    "summary":_summary([
                        record["median_geometric_distance"]/scale
                        for record in ideals[str(size)]
                    ]),
                } for size in ideal_sizes
            },
            "illustrative_epsilon_for_ecdf":illustrative_epsilon,
            "sigma_fit_summary":_summary(sigma_values),
            "controller_median_summary":_summary(controller_values),
            "independent_null_median_summary":_summary(final_medians),
            "independent_null_fraction_within_controller_band":float(
                np.mean(np.abs(np.asarray(final_medians)-2.)<.1)),
            "corrected_null_ecdf_p10_summary":_summary(quantiles10),
            "corrected_null_ecdf_p50_summary":_summary(quantiles50),
            "corrected_null_ecdf_p90_summary":_summary(quantiles90),
            "trials":per_fit,
        }
    return {
        "status":"NONCERTIFIED_DPF5_TWO_REFERENCE_STABILITY_PROTOCOL",
        "mop":"DPF5","M":3,"D":2,
        "frozen_baseline_version":original.get("version"),
        "frozen_baseline_sha256":hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "frozen_GT_sha256":_sha(gt),
        "frozen_GT_count":len(gt),
        "archived_GT_exact_100_by_100_unique_PF":True,
        "ideal_law":"iid_uniform_decision_parameters_x1_x2_no_objective_dedup",
        "ideal_branch_D_theoretical_mass":2/3,
        "failure_law":"one_outward_half_normal_per_distinct_frozen_GT_anchor",
        "failure_branch_D_anchor_mass":67/3367,
        "sigma_target_normalized_distance":2.,
        "sigma_controller_tolerance_relative":.05,
        "sigma_controller_iterations_max":6,
        "ecdf_storage_order_statistics":N_SAMPLES,
        "ideal_sizes":list(ideal_sizes),
        "ideal_seeds":list(ideal_seeds),
        "fit_seeds":list(fit_seeds),
        "blur_seeds":list(blur_seeds),
        "ideal_samples_by_size":ideals,
        "results_by_K":by_K,
        "no_normative_release":True,
        "no_baseline_or_GT_or_audit_report_modified":True,
        "no_statistical_acceptance_threshold_invented":True,
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--K",type=int,action="append",dest="ks")
    p.add_argument("--ideal-size",type=int,action="append",dest="ideal_sizes")
    p.add_argument("--ideal-seed",type=int,action="append",dest="ideal_seeds")
    p.add_argument("--fit-seed",type=int,action="append",dest="fit_seeds")
    p.add_argument("--blur-seed",type=int,action="append",dest="blur_seeds")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    dest=a.output.resolve()
    if (dest.exists() or dest.is_relative_to(RESOURCE_DIR)
            or dest.is_relative_to(REPORTS) or dest==OFFICIAL):
        p.error("Refusing overwrite or official scientific output location")
    report=study(
        ks=tuple(a.ks or (40,100,200)),
        ideal_sizes=tuple(a.ideal_sizes or (4096,16384)),
        ideal_seeds=tuple(a.ideal_seeds or (123,124,125,126)),
        fit_seeds=tuple(a.fit_seeds or (1100,1101,1102,1103)),
        blur_seeds=tuple(a.blur_seeds or (1600,1601,1602,1603)),
    )
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print("NONCERTIFIED DPF5 dual-reference protocol:",report["frozen_GT_count"],
          "GT points, K=",sorted(report["results_by_K"]))
    for k,row in report["results_by_K"].items():
        print("K",k,
              "epsilon",row["ideal_epsilon_by_size"][
                  str(max(report["ideal_sizes"]))]["summary"],
              "sigma",row["sigma_fit_summary"],
              "independent null median",row["independent_null_median_summary"],
              "null fraction within 5% controller band",
              row["independent_null_fraction_within_controller_band"],
              "candidate ECDF p50",row["corrected_null_ecdf_p50_summary"])
    print("Not a released baseline; official GT, ECDF and audit report unchanged")

if __name__=="__main__":
    main()
