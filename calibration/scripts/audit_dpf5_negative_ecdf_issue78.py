#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only DPF5 Closeness negative-ECDF branch-weight audit.

The frozen GT is exactly unique(normalized(DPF5.pf(10000))).
The archived GT-normal blur uses each UNIQUE GT row once; its source
branch weights are 3300:67. In contrast the analytical decision-grid
pushforward uses the 100x100 original row multiplicities 3300:6700.

Hold the same GT, frozen sigma, s_K, per-GT normal and blurred point
fixed. Reweight ONLY the same blurred distances to isolate anchor
distribution effects. The result is a controlled COUNTERFACTUAL and
does not imply that one null distribution is normative or that the
archived ECDF is regenerated exactly by the current code.

Optionally run two genuine NSGA-II experiments against Closeness-only
in-memory baselines with the same independently sampled continuous
ideal-residue epsilon, but different null ECDFs. All other FAIR and
Q metrics, P0, Pf, GT, and s_K are checked unchanged.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np
from scipy.spatial import cKDTree

import moeabench as mb
from moeabench.diagnostics import baselines
from calibration.scripts.generate_baselines import (
    estimate_gt_normals, generate_half_normal_blur, SEED_START
)
from calibration.scripts.study_dpf5_independent_pf_issue78 import (
    FROZEN, PACKAGE, RESOURCES, REPORTS, OFFICIAL,
    fresh_dpf5_pf
)


def weighted_quantiles(values, weights, probabilities):
    """Exact empirical weighted lower quantiles for deterministic ECDF."""
    v=np.asarray(values,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64)
    p=np.asarray(probabilities,dtype=np.float64)
    if (v.ndim!=1 or w.shape!=v.shape or not len(v) or
            np.any(~np.isfinite(v)) or np.any(~np.isfinite(w)) or
            np.any(w<0) or not np.any(w>0) or
            np.any(~np.isfinite(p)) or np.any((p<0)|(p>1))):
        raise ValueError("Invalid quantiles, weights or distances")
    order=np.argsort(v,kind="stable")
    sorted_v=v[order]
    cumulative=np.cumsum(w[order])/float(np.sum(w))
    indexes=np.searchsorted(cumulative,p,side="left")
    return sorted_v[np.clip(indexes,0,len(sorted_v)-1)]


def frozen_dpf5_counts(gt,ideal,nadir):
    """Map original 10k decision-grid rows to exact frozen unique GT rows."""
    mop=mb.mops.DPF5(D=2,M=3)
    X=np.asarray(mop.ps(n_points=10000),dtype=np.float64)
    F=np.asarray(mop.evaluation(X)["F"],dtype=np.float64)
    if len(X)!=10000 or F.shape!=(10000,3):
        raise AssertionError("The original DPF5 source grid changed")
    normalized=(F-ideal)/(nadir-ideal)
    distance,indices=cKDTree(gt).query(normalized,k=1)
    if np.max(distance)>1e-12:
        raise AssertionError("Frozen DPF5 GT does not reproduce original objective support")
    raw=np.bincount(indices,minlength=len(gt)).astype(np.int64)
    D=X[:,0]>1/3
    Dcounts=np.bincount(indices[D],minlength=len(gt)).astype(np.int64)
    Pcounts=np.bincount(indices[~D],minlength=len(gt)).astype(np.int64)
    if (not np.all(raw==Dcounts+Pcounts) or not np.all(raw>0) or
            np.any((Dcounts>0)&(Pcounts>0))):
        raise AssertionError("Unexpected branch overlap or missing frozen GT rows")
    return raw,Pcounts,Dcounts


def study(*,K=100,reference_seed=123,reference_n=16384,
          optimizer_seeds=(1,7),generations=300,paired=True):
    if K<8 or reference_n<64 or len(set(optimizer_seeds))!=len(optimizer_seeds):
        raise ValueError("Invalid K/reference/optimizer seeds")
    source=json.loads(FROZEN.read_text())
    gt=np.asarray(source["problems"]["DPF5"]["gt_reference"],dtype=np.float64)
    archived_cl=source["problems"]["DPF5"][str(K)]["closeness"]
    with np.load(PACKAGE,allow_pickle=False) as pkg:
        if not np.array_equal(gt,np.asarray(pkg["gt_norm"])):
            raise ValueError("Frozen GT package mismatch")
        ideal=np.asarray(pkg["ideal"],dtype=np.float64)
        nadir=np.asarray(pkg["nadir"],dtype=np.float64)
    raw_counts,Pcounts,Dcounts=frozen_dpf5_counts(gt,ideal,nadir)
    normal=estimate_gt_normals(gt)
    sigma=float(archived_cl["sigma"])
    scale=float(baselines.get_resolution_factor_k(gt,K,seed=0))
    if not np.isfinite(sigma) or sigma<=0 or scale<=0:
        raise AssertionError("Invalid archived sigma or s_K")
    blur_seed=SEED_START+K+500
    blurred=generate_half_normal_blur(
        gt,normal,sigma,np.random.RandomState(blur_seed))
    raw_u=cKDTree(gt).query(blurred,k=1)[0]/scale
    reference,ref_D=fresh_dpf5_pf(reference_n,seed=reference_seed)
    new_ideal=(reference-ideal)/(nadir-ideal)
    if np.min(cKDTree(gt).query(new_ideal,k=1)[0])<=1e-10:
        raise AssertionError("Analytical reference is not independent of frozen GT")
    epsilon=float(np.median(cKDTree(gt).query(new_ideal,k=1)[0])/scale)
    if not epsilon>0:
        raise AssertionError("Missing positive independent ideal residue")
    grid=np.linspace(0.0,1.0,200)
    weights={
        "distinct_frozen_GT_rows":np.ones(len(gt),dtype=np.float64),
        "original_decision_grid_multiplicity":raw_counts.astype(np.float64),
    }
    nulls={}
    candidates={}
    for name,w in weights.items():
        quantiles=weighted_quantiles(raw_u,w,grid)
        corrected=np.maximum(0.0,quantiles-epsilon)
        record=dict(archived_cl)
        record.update({
            "ideal_res":epsilon,
            "ideal_res_method":"NONCERTIFIED_DPF5_iid_decision_parameters",
            "rand_ecdf":corrected.tolist(),
            "rand50":float(np.maximum(0.0,weighted_quantiles(raw_u,w,[0.5])[0]-epsilon)),
            "p95":float(np.maximum(0.0,weighted_quantiles(raw_u,w,[.95])[0]-epsilon)),
        })
        cand=copy.deepcopy(source)
        cand["problems"]["DPF5"][str(K)]["closeness"]=record
        candidates[name]=cand
        nulls[name]={
            "branch_D_anchor_probability":float(
                np.sum(w[Dcounts>0])/np.sum(w)),
            "raw_blur_distance_median":float(weighted_quantiles(raw_u,w,[.5])[0]),
            "raw_blur_distance_p95":float(weighted_quantiles(raw_u,w,[.95])[0]),
            "corrected_rand_ecdf_sha256":hashlib.sha256(
                np.ascontiguousarray(corrected).tobytes()).hexdigest(),
            "corrected_rand50":record["rand50"],
            "corrected_rand_ecdf_length":len(corrected),
        }
    archived_ecdf=np.asarray(archived_cl["rand_ecdf"],dtype=float)
    report={
        "status":"NONCERTIFIED_DPF5_NULL_ECDF_WEIGHTING_AUDIT",
        "frozen_calibration_version":source.get("version"),
        "frozen_GT_points":len(gt),
        "frozen_GT_sha256":hashlib.sha256(
            np.ascontiguousarray(gt).tobytes()).hexdigest(),
        "original_decision_grid_rows":int(np.sum(raw_counts)),
        "distinct_P_rows":int(np.count_nonzero(Pcounts)),
        "distinct_D_rows":int(np.count_nonzero(Dcounts)),
        "original_P_decision_rows":int(np.sum(Pcounts)),
        "original_D_decision_rows":int(np.sum(Dcounts)),
        "negative_null_mechanism":"One fixed half-normal GT-normal perturbation per distinct GT row; candidate CDFs differ ONLY in anchor row weights",
        "frozen_sigma":sigma,"frozen_s_K":scale,
        "blur_seed":int(blur_seed),
        "independent_ideal_sample_count":int(reference_n),
        "independent_ideal_seed":int(reference_seed),
        "independent_ideal_D_fraction":float(np.mean(ref_D)),
        "ideal_residue":epsilon,
        "null_ecdfs":nulls,
        "archived_null_ecdf_count":int(len(archived_ecdf)),
        "archived_null_ecdf_median":float(np.median(archived_ecdf)),
        "regenerated_distinct_null_matches_archived_with_tolerance_1e6":bool(
            len(archived_ecdf)==200 and
            np.max(np.abs(archived_ecdf-
                          np.asarray(candidates["distinct_frozen_GT_rows"]
                                     ["problems"]["DPF5"][str(K)]
                                     ["closeness"]["rand_ecdf"])))<1e-6),
        "paired_real_moea":[],
        "q_shift_by_null_weighting":None,
        "other_fair_metrics_unchanged":None,
        "scientific_measure_approved":False,
        "baseline_modified":False,
    }
    if paired:
        if len(optimizer_seeds)<2:
            raise ValueError("Require two independent optimizer seeds")
        for seed in optimizer_seeds:
            exp=mb.experiment(
                mop=mb.mops.DPF5(D=2,M=3),
                moea=mb.moeas.NSGA2(population=K,generations=generations))
            exp.run(seed=int(seed),silent=True)
            archived=mb.clinic.audit(exp)
            if archived.fr is None or archived.quality is None:
                raise AssertionError("Missing original clinical report")
            ctx=archived.diagnostic_context
            outcomes={}
            for name,cand in candidates.items():
                with baselines.use_baselines(cand):
                    altered=mb.clinic.audit(exp)
                if altered.fr is None or altered.quality is None:
                    raise AssertionError("Missing paired clinical report")
                for array in ("GT","P_initial","P_final"):
                    if not np.array_equal(ctx[array],altered.diagnostic_context[array]):
                        raise AssertionError("Paired optimizer or GT altered")
                if ctx["s_k"]!=altered.diagnostic_context["s_k"]:
                    raise AssertionError("Paired frozen scale changed")
                np.testing.assert_allclose(
                    altered.fr.metrics["CLOSENESS"].raw_data,
                    np.maximum(0.0,
                        archived.fr.metrics["CLOSENESS"].raw_data-epsilon),
                    rtol=0,atol=1e-12)
                for key in ("COVERAGE","GAP","REGULARITY","BALANCE","HEADWAY"):
                    np.testing.assert_allclose(
                        [archived.fr.metrics[key].value,
                         archived.quality.scores["Q_"+key].value],
                        [altered.fr.metrics[key].value,
                         altered.quality.scores["Q_"+key].value],
                        rtol=0,atol=1e-12)
                outcomes[name]=float(
                    altered.quality.scores["Q_CLOSENESS"].value)
            again=mb.clinic.audit(exp)
            np.testing.assert_allclose(
                again.quality.scores["Q_CLOSENESS"].value,
                archived.quality.scores["Q_CLOSENESS"].value,
                atol=1e-12,rtol=0)
            report["paired_real_moea"].append({
                "seed":int(seed),"generations":int(generations),
                "archived_q_closeness":float(
                    archived.quality.scores["Q_CLOSENESS"].value),
                "q_by_negative_null_weight":outcomes,
                "null_weighting_absolute_q_difference":abs(
                    outcomes["distinct_frozen_GT_rows"]-
                    outcomes["original_decision_grid_multiplicity"]),
            })
        report["q_shift_by_null_weighting"]=max(
            x["null_weighting_absolute_q_difference"]
            for x in report["paired_real_moea"])
        report["other_fair_metrics_unchanged"]=True
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--K",type=int,default=100)
    p.add_argument("--reference-n",type=int,default=16384)
    p.add_argument("--reference-seed",type=int,default=123)
    p.add_argument("--seed",type=int,action="append",dest="seeds")
    p.add_argument("--generations",type=int,default=300)
    p.add_argument("--skip-real-moea",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if (out.exists() or out.is_relative_to(RESOURCES)
            or out.is_relative_to(REPORTS) or out==OFFICIAL):
        p.error("Refusing existing or official output file")
    result=study(K=a.K,reference_n=a.reference_n,
                 reference_seed=a.reference_seed,
                 optimizer_seeds=tuple(a.seeds or (1,7)),
                 generations=a.generations,paired=not a.skip_real_moea)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({
        "status":result["status"],
        "ideal_residue":result["ideal_residue"],
        "negative_null_branch_mass":{
            k:v["branch_D_anchor_probability"]
            for k,v in result["null_ecdfs"].items()
        },
        "negative_null_raw_median":{
            k:v["raw_blur_distance_median"] for k,v in result["null_ecdfs"].items()
        },
        "q_shift_by_null_weighting":result["q_shift_by_null_weighting"],
        "real_moea":result["paired_real_moea"],
        "baseline_modified":result["baseline_modified"],
    },indent=2))


if __name__=="__main__":
    main()
