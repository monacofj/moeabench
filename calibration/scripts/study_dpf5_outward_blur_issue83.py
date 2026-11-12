#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only DPF5 study with corrected one-sided outward GT half-blur.

Synchronized source generator from MERGED PR #84. Original DPF5 frozen
GT, distinct-GT-row null-anchor law and original K-dependent s_K are
unchanged. Calibrate only sigma to the SAME median 2*s_K target,
generate a new outward-only half-normal failure ECDF, and combine it
with a separate iid decision-parameter analytical Pareto reference
residue (PROPOSED, not scientifically approved).

Compare original historical Q, old-null-with-only-ideal-residue
corrected, and corrected-outward-null-with-same-residue using EXACT
same NSGA-II P0/Pf/GT and all other FAIR/Q families fixed.

NEVER overwrite or install any baseline, GT, audit_report or candidate.
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
    estimate_gt_normals, generate_half_normal_blur,
    calibrate_sigma, SEED_START, N_SAMPLES,
)
from calibration.scripts.stage_closeness_adr0037 import revise_closeness_record
from calibration.scripts.study_dpf5_independent_pf_issue78 import (
    FROZEN, PACKAGE, RESOURCES, REPORTS, OFFICIAL, fresh_dpf5_pf,
)
from calibration.scripts.audit_dpf5_negative_ecdf_issue78 import (
    frozen_dpf5_counts,
)


def _hash(values):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(values,dtype=np.float64)).tobytes()).hexdigest()


def _in_memory_candidates(*,K=100,reference_n=16384,reference_seed=123):
    source=json.loads(FROZEN.read_text())
    gt=np.asarray(source["problems"]["DPF5"]["gt_reference"],dtype=float)
    old=source["problems"]["DPF5"][str(K)]["closeness"]
    with np.load(PACKAGE,allow_pickle=False) as pkg:
        if not np.array_equal(gt,np.asarray(pkg["gt_norm"],dtype=float)):
            raise ValueError("Frozen GT JSON/NPZ mismatch")
        ideal=np.asarray(pkg["ideal"],dtype=float)
        nadir=np.asarray(pkg["nadir"],dtype=float)
    raw_counts,P_counts,D_counts=frozen_dpf5_counts(gt,ideal,nadir)
    if int(np.sum(raw_counts))!=10000 or len(gt)!=3367:
        raise ValueError("Historic DPF5 original-grid GT support changed")
    scale=float(baselines.get_resolution_factor_k(gt,K,seed=0))
    tree=cKDTree(gt)
    normals=estimate_gt_normals(gt)
    sigma,median_controller,p95_controller=calibrate_sigma(
        gt,normals,scale,SEED_START+K,tree=tree)
    if not np.isfinite(sigma) or sigma<=0:
        raise ValueError("Invalid corrected half-blur sigma")
    # Source generator MUST enforce domination, not merely nonnegative F.
    blurred=generate_half_normal_blur(
        gt,normals,sigma,np.random.RandomState(SEED_START+K+500))
    if not (np.all(blurred>=gt) and
            np.all(np.any(blurred>gt,axis=1))):
        raise AssertionError("Corrected generator failed strict outward geometry")
    physical_dists=tree.query(blurred,k=1)[0]/scale
    median_observed=float(np.median(physical_dists))
    if abs(median_controller-2.)>=.1:
        raise AssertionError("Unchanged sigma 2*s_K target did not converge")
    if not 1.8<=median_observed<=2.2:
        raise AssertionError("Independent corrected blur seed is far from 2*s_K")

    fresh,labels=fresh_dpf5_pf(reference_n,seed=reference_seed)
    independent_norm=(fresh-ideal)/(nadir-ideal)
    distances=tree.query(independent_norm,k=1)[0]/scale
    if np.min(distances)<=1e-10:
        raise AssertionError("Fresh DPF5 ideal reference overlaps old GT")
    epsilon=float(np.median(distances))
    if not np.isfinite(epsilon) or epsilon<=0:
        raise AssertionError("No independent finite-GT epsilon")

    old_null=revise_closeness_record(old,epsilon)
    old_null["ideal_res_method"]="NONCERTIFIED_iid_PF_old_frozen_blur"
    new_null=dict(old)
    corrected_distances=np.maximum(0.,physical_dists-epsilon)
    sorted_null=np.sort(corrected_distances)
    if len(sorted_null)>N_SAMPLES:
        indices=np.linspace(0,len(sorted_null)-1,N_SAMPLES).astype(int)
        empirical=sorted_null[indices]
    else:
        empirical=sorted_null
    new_null.update({
        "ideal_res":epsilon,
        "ideal_res_method":"NONCERTIFIED_iid_DPF5_PF_OUTWARD_BLUR_83",
        "sigma":float(sigma),
        "rand_ecdf":empirical.tolist(),
        "rand50":float(np.median(corrected_distances)),
        "p95":float(np.percentile(corrected_distances,95)),
        "seed":SEED_START+K,
    })
    candidates={}
    for name,record in {
        "historical_null_only_ideal_residue_fixed":old_null,
        "corrected_one_sided_outward_null":new_null,
    }.items():
        candidate=copy.deepcopy(source)
        candidate["problems"]["DPF5"][str(K)]["closeness"]=record
        candidates[name]=candidate

    details={
        "status":"NONCERTIFIED_DPF5_CORRECTED_OUTWARD_BLUR_PAIRED_STUDY",
        "GT_sha256":_hash(gt),
        "frozen_baseline_sha256":hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "GT_distinct_points":len(gt),
        "GT_distinct_P":int(np.count_nonzero(P_counts)),
        "GT_distinct_D":int(np.count_nonzero(D_counts)),
        "GT_negative_null_D_anchor_fraction":float(
            np.count_nonzero(D_counts)/len(gt)),
        "GT_independent_ideal_D_theoretical_mass":2/3,
        "GT_independent_ideal_D_empirical_mass":float(np.mean(labels)),
        "K":int(K),"s_K":scale,
        "original_sigma":float(old["sigma"]),
        "new_sigma":float(sigma),
        "sigma_controller_target_median_u":2.,
        "sigma_controller_observed_median_u":float(median_controller),
        "independent_final_blur_median_u":median_observed,
        "independent_final_blur_p95_u":float(np.percentile(physical_dists,95)),
        "independent_ideal_residue":epsilon,
        "independent_PF_n":int(reference_n),
        "independent_PF_seed":int(reference_seed),
        "blur_objectives_sha256":_hash(blurred),
        "all_blurs_strictly_dominated_by_parent":True,
        "original_rand50":float(old["rand50"]),
        "new_outward_rand50":float(new_null["rand50"]),
        "original_null_ecdf_sha256":_hash(old["rand_ecdf"]),
        "new_outward_null_ecdf_sha256":_hash(new_null["rand_ecdf"]),
        "null_anchors_identical_distinct_GT_rows":True,
        "null_weighting_revised":False,
        "historical_GT_changed":False,
        "baseline_files_changed":False,
        "certified":False,
    }
    return source,candidates,details


def study(*,K=100,reference_n=16384,reference_seed=123,
          generations=300,optimizer_seeds=(1,7),paired=True):
    source,candidates,report=_in_memory_candidates(
        K=K,reference_n=reference_n,reference_seed=reference_seed)
    report["optimizer_generations"]=int(generations)
    report["paired_real_moea"]=[]
    report["all_other_fair_q_unchanged"]=None
    if paired:
        if len(set(optimizer_seeds))<2 or generations<2:
            raise ValueError("Need at least two optimizer seeds and genuine budget")
        for seed in optimizer_seeds:
            exp=mb.experiment(
                mop=mb.mops.DPF5(D=2,M=3),
                moea=mb.moeas.NSGA2(population=K,generations=generations))
            exp.run(seed=int(seed),silent=True)
            original=mb.clinic.audit(exp)
            if original.fr is None or original.quality is None:
                raise AssertionError("Clinical audit unavailable")
            context=original.diagnostic_context
            if context["gt_source"]!="registered":
                raise AssertionError("Expected registered unchanged historical GT")
            scores={}
            epsilon=report["independent_ideal_residue"]
            for key,candidate in candidates.items():
                with baselines.use_baselines(candidate):
                    new=mb.clinic.audit(exp)
                if new.fr is None or new.quality is None:
                    raise AssertionError("In-memory candidate failed clinical audit")
                new_ctx=new.diagnostic_context
                for arr in ("GT","P_initial","P_final"):
                    if not np.array_equal(context[arr],new_ctx[arr]):
                        raise AssertionError("Changed paired scientific "+arr)
                if context["s_k"]!=new_ctx["s_k"]:
                    raise AssertionError("Reference scale changed")
                np.testing.assert_allclose(
                    new.fr.metrics["CLOSENESS"].raw_data,
                    np.maximum(0.,original.fr.metrics["CLOSENESS"].raw_data-epsilon),
                    atol=1e-12,rtol=0)
                for m in ("HEADWAY","COVERAGE","GAP","REGULARITY","BALANCE"):
                    np.testing.assert_allclose(
                        [new.fr.metrics[m].value,
                         new.quality.scores["Q_"+m].value],
                        [original.fr.metrics[m].value,
                         original.quality.scores["Q_"+m].value],
                        atol=1e-12,rtol=0)
                scores[key]=float(new.quality.scores["Q_CLOSENESS"].value)
            original_again=mb.clinic.audit(exp)
            np.testing.assert_allclose(
                original_again.quality.scores["Q_CLOSENESS"].value,
                original.quality.scores["Q_CLOSENESS"].value,
                atol=1e-12,rtol=0)
            report["paired_real_moea"].append({
                "optimizer_seed":int(seed),
                "P0_sha256":_hash(context["P_initial"]),
                "Pf_sha256":_hash(context["P_final"]),
                "registered_GT_sha256":_hash(context["GT"]),
                "original_historical_Q_CLOSENESS":float(
                    original.quality.scores["Q_CLOSENESS"].value),
                "Q_by_candidate":scores,
                "outward_null_vs_old_null_Q_difference":float(
                    scores["corrected_one_sided_outward_null"]-
                    scores["historical_null_only_ideal_residue_fixed"]),
            })
        report["all_other_fair_q_unchanged"]=True
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--K",type=int,default=100)
    p.add_argument("--reference-n",type=int,default=16384)
    p.add_argument("--reference-seed",type=int,default=123)
    p.add_argument("--generations",type=int,default=300)
    p.add_argument("--seed",type=int,action="append",dest="seeds")
    p.add_argument("--skip-real-moea",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    dest=a.output.resolve()
    if dest.exists() or dest.is_relative_to(RESOURCES) or dest.is_relative_to(REPORTS) or dest==OFFICIAL:
        p.error("No overwriting existing or official scientific data")
    result=study(K=a.K,reference_n=a.reference_n,
                 reference_seed=a.reference_seed,
                 generations=a.generations,
                 optimizer_seeds=tuple(a.seeds or (1,7)),
                 paired=not a.skip_real_moea)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps({
        "status":result["status"],
        "K":result["K"],
        "s_K":result["s_K"],
        "original_sigma":result["original_sigma"],
        "new_sigma":result["new_sigma"],
        "sigma_controller_observed_median_u":result["sigma_controller_observed_median_u"],
        "independent_final_blur_median_u":result["independent_final_blur_median_u"],
        "ideal_residue":result["independent_ideal_residue"],
        "new_outward_rand50":result["new_outward_rand50"],
        "Q_real_MOEA":result["paired_real_moea"],
        "certified":result["certified"],
    },indent=2))


if __name__=="__main__":
    main()
