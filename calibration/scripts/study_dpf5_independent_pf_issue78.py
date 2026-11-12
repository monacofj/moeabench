#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent DPF5 reference by continuous Pareto-set parameter pushforward.

PROPOSAL ONLY: iid x1,x2 ~ Uniform[0,1], x3=x1, trailing decision
variables = 0.5. Under the actual DPF5 M3/D2 objective mapping, this
pushforward assigns P mass 1/3 and D mass 2/3 without deduplicating
objective points or introducing a new objective-area law.

Use the exact unchanged v0.14.0 GT (deduplicated original 100×100
decision grid), its frozen physical normalization and s_K. Do not
create, modify or certify any baseline, reference package or report.
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
from moeabench.diagnostics import baselines
from moeabench.diagnostics.calibration import estimate_ideal_residue

from calibration.scripts.study_dpf5_branch_measure_issue78 import (
    FROZEN, PACKAGE, RESOURCES, REPORTS, OFFICIAL, _sha
)


def fresh_dpf5_pf(n_points, *, seed, mop=None):
    """Return independent objective samples and true branch labels."""
    if not isinstance(n_points,int) or n_points<64 or n_points>200000:
        raise ValueError("Expected bounded n_points in [64,200000]")
    if not isinstance(seed,int) or seed<0:
        raise ValueError("Expected nonnegative integer seed")
    mop=mop or DPF5(D=2,M=3)
    if not (mop.M==3 and mop.D==2):
        raise ValueError("This proposal is ONLY for DPF5 M3/D2")
    rng=np.random.default_rng(seed)
    xy=rng.random((n_points,2))
    X=np.zeros((n_points,mop.N),dtype=float)
    X[:,:2]=xy
    X[:,2]=xy[:,0]
    X[:,3:]=0.5
    F=np.asarray(mop.evaluation(X)["F"],dtype=float)
    if F.shape!=(n_points,3) or not np.all(np.isfinite(F)):
        raise AssertionError("Invalid DPF5 analytical front")
    # On the nonnegative unit sphere, no two distinct points can
    # componentwise dominate each other. This is a structural
    # argument rather than an expensive O(n^2) sample-only check.
    if np.min(F)<-1e-12 or np.max(np.abs(np.linalg.norm(F,axis=1)-1))>1e-12:
        raise AssertionError("Independently generated objectives not on positive unit sphere")
    return F, xy[:,0]>1/3


def run_study(*,n_points=(4096,16384),seeds=(123,124,125,126),
              ks=(40,100,200)):
    n_points=tuple(n_points)
    seeds=tuple(seeds)
    ks=tuple(ks)
    if (not n_points or len(set(n_points))!=len(n_points) or
            not seeds or len(set(seeds))!=len(seeds) or
            not ks or len(set(ks))!=len(ks) or any(k<2 for k in ks)):
        raise ValueError("Require distinct sizes/seeds/K")
    source=json.loads(FROZEN.read_text())
    gt=np.asarray(source["problems"]["DPF5"]["gt_reference"],dtype=float)
    with np.load(PACKAGE,allow_pickle=False) as package:
        frozen=np.asarray(package["gt_norm"],dtype=float)
        ideal=np.asarray(package["ideal"],dtype=float)
        nadir=np.asarray(package["nadir"],dtype=float)
    if not np.array_equal(gt,frozen) or np.any(nadir<=ideal):
        raise ValueError("Frozen GT and normalization package mismatch")
    mop=DPF5(D=2,M=3)
    original_grid=np.asarray(mop.pf(n_points=10000),dtype=float)
    candidate_support=np.unique((original_grid-ideal)/(nadir-ideal),axis=0)
    if (len(candidate_support)!=len(gt) or
            not np.allclose(candidate_support,np.unique(gt,axis=0),
                            rtol=0,atol=1e-12)):
        raise ValueError("Independent law unsafe: frozen GT does not match deduplicated 100x100 source")
    scales={str(k):float(baselines.get_resolution_factor_k(gt,k,seed=0))
            for k in ks}
    if not all(np.isfinite(x) and x>0 for x in scales.values()):
        raise AssertionError("Invalid resolution scale")
    rows=[]
    for N in n_points:
        for seed in seeds:
            physical,D=fresh_dpf5_pf(N,seed=seed,mop=mop)
            normalized=(physical-ideal)/(nadir-ideal)
            # Use exactly the independent GT/residue guard used by
            # the supported ADR0037 calibration, not a hand-tuned law.
            distance=cKDTree(gt).query(normalized,k=1)[0]
            if not np.all(np.isfinite(distance)) or not np.all(distance>1e-10):
                raise AssertionError("Fresh ideal PF overlaps the frozen GT")
            eps={str(k):estimate_ideal_residue(
                gt,normalized,sk) for k,sk in scales.items()}
            rows.append({
                "n_points":int(N),"seed":int(seed),
                "independent_pf_sha256":_sha(normalized),
                "true_D_branch_fraction":float(np.mean(D)),
                "theoretical_D_branch_probability":2/3,
                "true_P_branch_fraction":float(np.mean(~D)),
                "raw_dist_median":float(np.median(distance)),
                "raw_dist_p95":float(np.percentile(distance,95)),
                "ideal_res":eps,
                "unique_objective_count":int(len(np.unique(normalized,axis=0))),
            })
    per_size={}
    for N in n_points:
        subset=[r for r in rows if r["n_points"]==N]
        per_size[str(N)]={
            "branch_D_fraction_min":min(x["true_D_branch_fraction"] for x in subset),
            "branch_D_fraction_max":max(x["true_D_branch_fraction"] for x in subset),
            "epsilon_seed_range":{
                str(k):{
                    "minimum":min(x["ideal_res"][str(k)] for x in subset),
                    "maximum":max(x["ideal_res"][str(k)] for x in subset),
                    "span":max(x["ideal_res"][str(k)] for x in subset)-
                           min(x["ideal_res"][str(k)] for x in subset),
                } for k in ks
            },
        }
    return {
        "status":"NONCERTIFIED_DPF5_IID_DECISION_PARAMETER_PF_PROPOSAL",
        "normative_measure_adopted":False,
        "sampling_rule":"iid_uniform_x1_x2_then_DPF5_M3D2_analytical_objectives",
        "analytical_branch_P_mass":1/3,
        "analytical_branch_D_mass":2/3,
        "area_uniformity_claimed":False,
        "archived_GT_exact_unique_100_grid_support_reproduced":True,
        "archived_GT_sha256":_sha(gt),
        "archived_GT_points":int(len(gt)),
        "frozen_baseline_sha256":hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "fixed_s_K":scales,
        "rows":rows,
        "summary_by_sample_size":per_size,
        "baseline_or_GT_modified":False,
        "certified":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points",action="append",type=int,dest="sizes")
    p.add_argument("--seed",action="append",type=int,dest="seeds")
    p.add_argument("--K",action="append",type=int,dest="ks")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if (out.exists() or out.is_relative_to(RESOURCES) or
            out.is_relative_to(REPORTS) or out==OFFICIAL):
        p.error("Refusing existing or official scientific output path")
    report=run_study(
        n_points=tuple(a.sizes or (4096,16384)),
        seeds=tuple(a.seeds or (123,124,125,126)),
        ks=tuple(a.ks or (40,100,200)),
    )
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    for n,data in report["summary_by_sample_size"].items():
        print("INDEPENDENT DPF5 n",n,
              "D-mass range",data["branch_D_fraction_min"],
              data["branch_D_fraction_max"])
        print("  K100 epsilon",data["epsilon_seed_range"].get("100",{}))
    print("NONCERTIFIED: no calibration/baseline changed")


if __name__=="__main__":
    main()
