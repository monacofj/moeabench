#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scientific challenge: does ideal_res depend on PF reservoir cardinality?

For each frozen GT, hold the requested final ideal sample size (1024), GT,
K and sampling seeds fixed while changing ONLY the analytical PF request's
oversampling factor. Deterministic PF implementations can change grid
resolution/distribution when n_points changes. An unbiased random selection
from one reservoir does NOT solve this deeper source-distribution dependency.

Diagnostic only: no baseline or canonical report writes; no invented
stability acceptance threshold and no claim of independent PF-generator RNG.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree

from moeabench import mops
from moeabench.diagnostics import baselines
from moeabench.diagnostics.calibration import estimate_ideal_residue
from calibration.scripts.generate_baselines import _analytical_ideal_from_package
from calibration.scripts.study_ideal_residue_robustness import _region_hist, _stats


def compare_reservoir_factors(gt, sampler, *, n_fresh=1024,
                              factors=(2,4,8,16), seeds=(123,124,125),
                              k_values=(20,100)):
    gt = np.asarray(gt,dtype=float)
    if gt.ndim != 2 or not np.all(np.isfinite(gt)) or len(gt) < max(k_values):
        raise ValueError("Invalid GT or unsupported K")
    if n_fresh < 2 or not factors or min(factors) < 1 or not seeds:
        raise ValueError("Invalid reservoir factors, requested sample, or seeds")
    tree = cKDTree(gt)
    thresh = max(1e-12, 1e-10*float(np.max(np.ptp(gt, axis=0))))
    scales = {str(k): float(baselines.get_resolution_factor_k(
        gt,k,seed=0)) for k in k_values}
    record = {"method": "same_GT_and_requested_sample_different_pf_cardinalities",
              "K_scales": scales, "n_fresh": n_fresh,
              "factors": {}, "all_factors_summary": {}}
    for factor in factors:
        count = len(gt) + factor*n_fresh
        candidates = np.asarray(sampler.pf(n_points=count),dtype=float)
        if (candidates.ndim != 2 or candidates.shape[1] != gt.shape[1] or
                not np.all(np.isfinite(candidates))):
            raise ValueError(f"Malformed analytic PF for factor {factor}")
        distances = tree.query(candidates,k=1)[0]
        fresh = np.unique(candidates[distances > thresh],axis=0)
        if len(fresh) < n_fresh:
            raise ValueError(f"factor {factor}: only {len(fresh)} unique fresh points")
        values=[]
        for seed in seeds:
            indices=np.random.default_rng(seed).choice(len(fresh),
                                                        size=n_fresh,replace=False)
            pts=fresh[indices]
            d=tree.query(pts,k=1)[0]
            eps={k:float(np.median(d)/s) for k,s in scales.items()}
            assert all(np.isfinite(v) and v > 0 for v in eps.values())
            values.append({"seed":seed,"epsilons":eps,
                           "regions_f1":_region_hist(pts,gt)})
        record["factors"][str(factor)]={
            "requested_pf_points":count,
            "returned_pf_points":len(candidates),
            "unique_fresh_candidates":len(fresh),
            "raw_non_GT_candidates": int(np.count_nonzero(distances > thresh)),
            "duplicate_non_GT_count": int(np.count_nonzero(distances > thresh) - len(fresh)),
            "duplicate_non_GT_fraction": float(
                1.0 - len(fresh)/max(1, np.count_nonzero(distances > thresh))
            ),
            "candidate_regions_f1":_region_hist(fresh,gt),
            "replicates":values,
            "epsilon_stats":{k:_stats([r["epsilons"][k] for r in values])
                             for k in scales},
        }
    for k in scales:
        all_eps=[v["epsilons"][k] for f in record["factors"].values()
                 for v in f["replicates"]]
        factor_medians={f:rec["epsilon_stats"][k]["median"]
                        for f,rec in record["factors"].items()}
        record["all_factors_summary"][k]={
            "all_replicate_range":_stats(all_eps),
            "factor_medians":factor_medians,
            "factor_median_range":_stats(list(factor_medians.values())),
        }
    return record


def analyze(name, frozen, *, factors=(2,4,8,16),
            seeds=(123,124,125), n_fresh=1024):
    if name not in frozen["problems"]:
        raise ValueError(f"Unknown baseline: {name}")
    path=ROOT/"moeabench/diagnostics/resources/references"/name/"calibration_package.npz"
    with np.load(path,allow_pickle=False) as pkg:
        gt=np.asarray(pkg["gt_norm"],dtype=float)
        if not np.array_equal(gt,np.asarray(frozen["problems"][name]["gt_reference"])):
            raise ValueError(f"{name}: baseline GT != packaged GT")
        if _analytical_ideal_from_package(name,gt,pkg) is None:
            raise ValueError(f"{name}: missing compatible independent PF")
        ideal=np.asarray(pkg["ideal"],dtype=float)
        span=np.asarray(pkg["nadir"],dtype=float)-ideal

    class NormalizedPF:
        def pf(self,n_points):
            return (np.asarray(getattr(mops,name)().pf(n_points=n_points),
                               dtype=float)-ideal)/span

    record=compare_reservoir_factors(gt,NormalizedPF(),
                    n_fresh=n_fresh,factors=factors,seeds=seeds)
    record["GT_sha256"]=hashlib.sha256(
        np.ascontiguousarray(gt).tobytes()).hexdigest()
    record["problem"]=name
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem",action="append",dest="problems",required=True)
    p.add_argument("--factor",action="append",dest="factors",type=int)
    p.add_argument("--seed",action="append",dest="seeds",type=int)
    p.add_argument("--n-fresh",type=int,default=1024)
    p.add_argument("--output",type=Path,required=True)
    opts=p.parse_args()
    output=opts.output.resolve()
    if output.exists():
        p.error("Refusing to overwrite prior scientific result")
    if output.is_relative_to((ROOT/"calibration/reports").resolve()) or output.is_relative_to(
        (ROOT/"moeabench/diagnostics/resources").resolve()
    ):
        p.error("Cannot write experiment into certified artifact directories")
    with (ROOT/"moeabench/diagnostics/resources/baselines_v0.14.0.json").open() as f:
        frozen=json.load(f)
    data={"status":"NONCERTIFIED_RESERVOIR_CARDINALITY_SENSITIVITY",
          "results":{},"failures":{}}
    for name in opts.problems:
        try:
            data["results"][name]=analyze(name,frozen,
                      factors=tuple(opts.factors or (2,4,8,16)),
                      seeds=tuple(opts.seeds or (123,124,125)),
                      n_fresh=opts.n_fresh)
            rec=data["results"][name]
            print(json.dumps({"problem":name,"K100_factor_medians":
                    rec["all_factors_summary"]["100"]["factor_medians"],
                    "K100_factor_relative_range":
                    rec["all_factors_summary"]["100"]["factor_median_range"]
                    ["relative_range_over_median"],
                    "candidate_regions_f1":{
                        factor:val["candidate_regions_f1"]
                        for factor,val in rec["factors"].items()},
                    "candidate_duplicate_fraction":{
                        factor:val["duplicate_non_GT_fraction"]
                        for factor,val in rec["factors"].items()} }),flush=True)
        except (ValueError,TypeError,OSError,IndexError) as exc:
            data["failures"][name]=f"{type(exc).__name__}: {exc}"
            print("UNSUPPORTED",name,data["failures"][name],flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    if data["failures"]:
        raise SystemExit("Some reservoir sensitivity experiments failed; inspect JSON")


if __name__=="__main__":
    main()
