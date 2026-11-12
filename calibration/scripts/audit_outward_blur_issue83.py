#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only, bounded 14-MOP outward GT-blur and 2*s_K controller audit.

Uses exact registered frozen normalized GT coordinates. A deterministic
512-row subset per MOP controls runtime. This verifies the direction
contract (blurred >= original for minimization), describes how much
local PCA normals need positive-cone projection, and measures the
unchanged six-iteration sigma controller's target error. This does not
make a new baseline, GT, certified Q score, or audit_report.
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
from moeabench.diagnostics.baselines import get_resolution_factor_k
from calibration.scripts.generate_baselines import (
    estimate_gt_normals, generate_half_normal_blur, calibrate_sigma,
    dominance_outward_direction,
)

RESOURCES=(ROOT/"moeabench/diagnostics/resources").resolve()
PACKAGES=RESOURCES/"references"
REPORTS=(ROOT/"calibration/reports").resolve()
OFFICIAL=(ROOT/"calibration/audit_report.html").resolve()
BASELINE=RESOURCES/"baselines_v0.14.0.json"


def _hash(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(
        a,dtype=np.float64)).tobytes()).hexdigest()


def evaluate_one(name, gt, *, max_points=512,K=100):
    gt=np.asarray(gt,dtype=float)
    if (gt.ndim!=2 or len(gt)<64 or gt.shape[1]<2 or
            not np.all(np.isfinite(gt)) or np.min(gt)<0):
        raise ValueError("Expected supported nonnegative archived reference")
    indices=np.unique(np.linspace(0,len(gt)-1,
                                  min(max_points,len(gt))).astype(int))
    subset=gt[indices]
    normals=estimate_gt_normals(subset)
    cosines=[]
    subspace_alignments=[]
    rescued_in_normal_plane=0
    low_rank_locations=0
    projections=0
    neighbors=min(len(subset)-1,int(np.clip(
        np.round(np.sqrt(len(subset))),10,30)))
    tree=cKDTree(subset)
    _,nearest=tree.query(subset,k=neighbors+1)
    for i,x in enumerate(subset):
        neighbors_points=subset[nearest[i,1:]]
        center=neighbors_points-np.mean(neighbors_points,axis=0)
        values,vec=np.linalg.eigh(center.T@center)
        n=vec[:,0]
        # A single PCA eigenvector may have arbitrary sign/mixed
        # components, especially on lower-dimensional PF sets.
        compatible=bool(np.all(n>=-1e-12) or np.all(n<=1e-12))
        projections+=int(not compatible)
        cosines.append(float(abs(np.dot(n,normals[i]))))
        threshold=max(float(values[-1]),0.0)*.02
        low=values<=threshold
        low[-1]=False
        low_rank=bool(np.count_nonzero(low)>=2)
        low_rank_locations+=int(low_rank)
        basis=vec[:,low] if low_rank else vec[:,:1]
        alignment=float(np.linalg.norm(basis.T@normals[i]))
        subspace_alignments.append(alignment)
        rescued_in_normal_plane+=int(
            not compatible and low_rank and alignment>1-1e-9)
    scale=float(get_resolution_factor_k(subset,K,seed=0))
    if not np.isfinite(scale) or scale<=0:
        raise AssertionError("Invalid registered K-resolution")
    sigma,median_fit,p95_fit=calibrate_sigma(
        subset,normals,scale,rng_seed=43083)
    actual=generate_half_normal_blur(
        subset,normals,sigma,np.random.RandomState(43883))
    distance=tree.query(actual,k=1)[0]/scale
    strict=np.any(actual>subset,axis=1)
    if (not np.all(actual>=subset) or
            not np.all(strict) or
            not np.all(np.isfinite(actual))):
        raise AssertionError(name+": outward dominated-side test failed")
    return {
        "MOP":name,
        "registered_GT_sha256":_hash(gt),
        "registered_GT_point_count":int(len(gt)),
        "bounded_subsample_count":int(len(subset)),
        "subsample_sha256":_hash(subset),
        "objective_count":int(gt.shape[1]),
        "K":int(K),
        "s_K_on_bounded_GT_subset":scale,
        "pca_normal_projection_required_count":int(projections),
        "pca_normal_projection_required_fraction":float(projections/len(subset)),
        "pca_normal_min_absolute_cosine_after_projection":float(min(cosines)),
        "pca_normal_median_absolute_cosine_after_projection":float(np.median(cosines)),
        "clear_low_dimensional_PCA_normal_subspace_count":int(low_rank_locations),
        "mixed_sign_eigenvector_rescued_by_true_normal_subspace":int(
            rescued_in_normal_plane),
        "median_alignment_to_estimated_normal_subspace":float(
            np.median(subspace_alignments)),
        "min_alignment_to_estimated_normal_subspace":float(
            min(subspace_alignments)),
        "blurs_weakly_dominated_by_parent_fraction":float(np.mean(np.all(actual>=subset,axis=1))),
        "blurs_strictly_dominated_by_parent_fraction":float(np.mean(strict)),
        "sigma":float(sigma),
        "calibration_controller_median_u":float(median_fit),
        "calibration_controller_target_abs_error":float(abs(median_fit-2.)),
        "calibration_controller_within_5pct":bool(abs(median_fit-2.)<0.1),
        "independent_blur_median_u":float(np.median(distance)),
        "independent_blur_p95_u":float(np.percentile(distance,95)),
        "NONCERTIFIED":True,
    }


def run(*,max_points=512,K=100):
    results={}
    source=json.loads(BASELINE.read_text())
    for name in sorted(source["problems"]):
        package=PACKAGES/name/"calibration_package.npz"
        if not package.exists():
            raise ValueError(name+": missing original GT package")
        with np.load(package,allow_pickle=False) as f:
            gt=np.asarray(f["gt_norm"],dtype=float)
        archived=np.asarray(source["problems"][name]["gt_reference"],dtype=float)
        if not np.array_equal(gt,archived):
            raise ValueError(name+": original frozen JSON/NPZ GT mismatch")
        results[name]=evaluate_one(name,gt,max_points=max_points,K=K)
    return {
        "status":"NONCERTIFIED_ISSUE83_OUTWARD_HALF_BLUR_AUDIT",
        "baseline_version":source.get("version"),
        "baseline_sha256":hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "same_sigma_target":2.0,
        "same_sigma_controller_iterations":6,
        "same_sigma_target_relative_tolerance":.05,
        "only_bounded_subsamples_audited":True,
        "MOP_count":len(results),
        "MOPs":results,
        "baseline_GT_and_audit_report_unchanged":True,
        "certified":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points",type=int,default=512)
    p.add_argument("--K",type=int,default=100)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    target=args.output.resolve()
    if target.exists() or target.is_relative_to(RESOURCES) or target.is_relative_to(REPORTS) or target==OFFICIAL:
        p.error("Refusing existing or official scientific output path")
    report=run(max_points=args.points,K=args.K)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+"\n")
    print("AUDIT",report["MOP_count"],"MOPs, K",args.K,
          "no official data modified")
    for name,row in report["MOPs"].items():
        print(name, "projection_fraction",
              f'{row["pca_normal_projection_required_fraction"]:.4f}',
              "subspace_recovered",row["mixed_sign_eigenvector_rescued_by_true_normal_subspace"],
              "subspace_alignment",f'{row["median_alignment_to_estimated_normal_subspace"]:.4f}',
              "calibration_median",f'{row["calibration_controller_median_u"]:.4f}',
              "inside_5percent",row["calibration_controller_within_5pct"],
              "outward",row["blurs_strictly_dominated_by_parent_fraction"])


if __name__=="__main__":
    main()
