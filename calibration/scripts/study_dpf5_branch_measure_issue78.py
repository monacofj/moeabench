#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Non-destructive DPF5 M3 conditional-branch Closeness-law sensitivity.

Compare *the same actual analytical objective points* under three empirical
measures: (1) uniform 2D decision grid pushforward, which retains repeated
objective points, (2) unique-objective counting, used implicitly when
independent_analytical_reference() discards duplicates, and (3) an EXPLICIT
illustrative 50:50 branch-stratified law. This isolates the effect of the
measure without changing GT, optimizer output, seed, noise or reference law.

The branch-stratified law is a *counterfactual*, not a scientific adoption.
The 2D intrinsic-area measure would assign zero probability to the
one-dimensional curve and is NOT the same as an equal-branch mixture.
No official baselines, reference packages, or audit_report are modified.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from moeabench.mops import DPF5
from moeabench.diagnostics import baselines

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "moeabench/diagnostics/resources/baselines_v0.14.0.json"
PACKAGE = ROOT / "moeabench/diagnostics/resources/references/DPF5/calibration_package.npz"
RESOURCES = (ROOT / "moeabench/diagnostics/resources").resolve()
REPORTS = (ROOT / "calibration/reports").resolve()
OFFICIAL = (ROOT / "calibration/audit_report.html").resolve()


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(
        a, dtype=np.float64)).tobytes()).hexdigest()


def weighted_median(values, weights):
    """Lower weighted empirical median, same convention for all three laws."""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if (len(values) != len(weights) or len(values) == 0 or
            np.any(~np.isfinite(values)) or np.any(~np.isfinite(weights)) or
            np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError("Invalid nonnegative finite weighted sample")
    order = np.argsort(values, kind="stable")
    index = np.searchsorted(np.cumsum(weights[order]),
                            0.5*float(np.sum(weights)), side="left")
    return float(values[order[min(index, len(values)-1)]])


def _archived_reference():
    source = json.loads(FROZEN.read_text())
    original = np.asarray(source["problems"]["DPF5"]["gt_reference"],dtype=float)
    with np.load(PACKAGE, allow_pickle=False) as package:
        frozen=np.asarray(package["gt_norm"],dtype=float)
        ideal=np.asarray(package["ideal"],dtype=float)
        nadir=np.asarray(package["nadir"],dtype=float)
    if (not np.array_equal(original,frozen) or not np.all(nadir>ideal)
            or original.shape[1]!=3):
        raise ValueError("Frozen DPF5 GT/normalization provenance does not match")
    return source,original,ideal,nadir


def compare_grid(side, *, ks=(40,100,200), reference=None):
    """Auditable paired probability-law comparison on one physical DPF5 grid."""
    if not isinstance(side,int) or side<8 or side>256:
        raise ValueError("Bounded decision-grid side must be [8,256]")
    if len(ks)==0 or any(not isinstance(k,int) or k<=0 for k in ks):
        raise ValueError("Require positive K values")
    source,gt,ideal,nadir = reference or _archived_reference()
    mop=DPF5(D=2,M=3)
    decision=np.asarray(mop.ps(n_points=side*side),dtype=float)
    if decision.shape[0]!=side*side:
        raise AssertionError("DPF5 analytical decision-grid cardinality changed")
    physical=np.asarray(mop.evaluation(decision)["F"],dtype=float)
    if physical.shape!=(side*side,3) or not np.all(np.isfinite(physical)):
        raise AssertionError("Invalid physical analytical PF")
    # In the actual DPF5 implementation with g=0, both pieces lie on
    # the nonnegative sphere, but the right branch collapses x2.
    radial=np.linalg.norm(physical,axis=1)
    if not np.allclose(radial,1.0,atol=1e-12,rtol=0):
        raise AssertionError("Analytical DPF5 source deviates from g=0 sphere")
    branch=decision[:,0] > (1.0/3.0)
    P,D=physical[~branch],physical[branch]
    unique_P=np.unique(P,axis=0)
    unique_D=np.unique(D,axis=0)
    if not len(unique_P) or not len(unique_D):
        raise AssertionError("One of DPF5's two branches is empty")
    # Use the original archived normalization and GT under EVERY law.
    tree=cKDTree(gt)
    normalize=lambda points:(points-ideal)/(nadir-ideal)
    raw_dist=tree.query(normalize(physical),k=1)[0]
    p_dist=tree.query(normalize(unique_P),k=1)[0]
    d_dist=tree.query(normalize(unique_D),k=1)[0]
    unique_dist=np.concatenate([p_dist,d_dist])
    weights_raw=np.ones(len(raw_dist),dtype=float)
    weights_unique=np.ones(len(unique_dist),dtype=float)
    # Explicitly impose 50:50 branch mass (illustration only), retaining
    # uniform conditional mass on the UNIQUE objectives of each branch.
    weights_50=np.r_[np.full(len(p_dist),.5/len(p_dist)),
                     np.full(len(d_dist),.5/len(d_dist))]
    med_raw=weighted_median(raw_dist,weights_raw)
    med_unique=weighted_median(unique_dist,weights_unique)
    med_half=weighted_median(unique_dist,weights_50)
    result={
        "status":"DPF5_MEASURE_SENSITIVITY_NOT_NORMATIVE",
        "grid_side":side,
        "source_law":"original_DPF5_parameter_grid",
        "grid_physical_objectives_sha256":_sha(physical),
        "frozen_GT_sha256":_sha(gt),
        "frozen_baseline_version":source.get("version"),
        "raw_objective_count":len(physical),
        "unique_P_count":len(unique_P),
        "unique_D_count":len(unique_D),
        "raw_branch_D_probability":float(np.mean(branch)),
        "dedup_branch_D_probability":float(len(unique_D)/(len(unique_P)+len(unique_D))),
        "illustrative_stratified_branch_D_probability":.5,
        "physical_PF_radial_max_error":float(np.max(np.abs(radial-1))),
        "measures":{
            "uniform_decision_grid_pushforward":{
                "median_distance_to_frozen_GT":med_raw,
                "branch_D_probability":float(np.mean(branch)),
            },
            "uniform_distinct_objective_vectors":{
                "median_distance_to_frozen_GT":med_unique,
                "branch_D_probability":float(len(unique_D)/(len(unique_P)+len(unique_D))),
            },
            "illustrative_equal_branch_mixture":{
                "median_distance_to_frozen_GT":med_half,
                "branch_D_probability":.5,
            },
        },
        "K":{},
    }
    for k in ks:
        scale=float(baselines.get_resolution_factor_k(gt,k,seed=0))
        if not np.isfinite(scale) or scale<=0:
            raise AssertionError("Invalid positive frozen s_K")
        eps={name:float(measure["median_distance_to_frozen_GT"]/scale)
             for name,measure in result["measures"].items()}
        result["K"][str(k)]={
            "s_K":scale,
            "ideal_res_by_measure":eps,
            "max_minus_min_ideal_res":float(max(eps.values())-min(eps.values())),
            "max_to_min_ideal_res_ratio":float(
                max(eps.values())/min(eps.values())
            ) if min(eps.values())>0 else None,
        }
    return result


def run(sides=(32,64,128),ks=(40,100,200)):
    if len(set(sides))!=len(sides) or not sides:
        raise ValueError("Grid sides must be unique")
    archive=_archived_reference()
    result={
        "status":"NONCERTIFIED_DPF5_MIXED_DIMENSION_MEASURE_AUDIT",
        "goal":"Isolate the sensitivity of ideal_res to sample weighting",
        "not_a_release_baseline":True,
        "objective_area_uniformity_adopted":False,
        "stratified_half_and_half_is_only_a_counterfactual":True,
        "frozen_baseline_sha256":hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "grids":{str(side):compare_grid(side,ks=ks,reference=archive)
                 for side in sides},
    }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side",type=int,action="append",dest="sides")
    parser.add_argument("--K",type=int,action="append",dest="ks")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    path=args.output.resolve()
    if (path.exists() or path.is_relative_to(RESOURCES) or
            path.is_relative_to(REPORTS) or path==OFFICIAL):
        parser.error("Refusing existing or official scientific destination")
    result=run(tuple(args.sides or (32,64,128)),tuple(args.ks or (40,100,200)))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    for side,row in result["grids"].items():
        print("GRID",side,"RAW-MASS",row["raw_branch_D_probability"],
              "DEDUP-MASS",row["dedup_branch_D_probability"])
        print("  medians",{
            k:round(v["median_distance_to_frozen_GT"],9)
            for k,v in row["measures"].items()
        })
        print("  epsilon K",{
            k:round(v["max_minus_min_ideal_res"],7) for k,v in row["K"].items()
        })
    print("NONCERTIFIED: no PF measure chosen, no baseline modified")


if __name__=="__main__":
    main()
