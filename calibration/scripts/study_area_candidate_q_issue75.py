#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #75: paired real-MOEA Q impact of an opt-in area-law baseline.

Run the SAME real NSGA-II experiment with the frozen v0.14.0 and a
provisionally area-corrected baseline. Keep GT, normalized fronts, K, s_K
and every non-Closeness FAIR metric fixed. Explicitly track Q clipping.
Neither experiment source nor certified calibration files are modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import moeabench as mb

from moeabench.diagnostics import baselines
from calibration.scripts.study_real_moea_q_sensitivity import (
    clinical_q_for_fixed_front, _digest,
)
from calibration.scripts.verify_closeness_migration import verify

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "moeabench/diagnostics/resources/baselines_v0.14.0.json"
PROTECTED = (ROOT / "moeabench/diagnostics/resources").resolve()
CANONICAL_REPORT = (ROOT / "calibration/audit_report.html").resolve()

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def score_fronts(name, candidate_path, *, optimizer_seeds=(1,7),
                 population=100, generations=500):
    """Score real fixed fronts, cross-check against full clinical final Q."""
    if name not in ("DTLZ1", "DTLZ2"):
        raise ValueError("Only independently sourced area-law M3 MOPs")
    if len(set(optimizer_seeds)) < 2 or generations < 4:
        raise ValueError("Need independent run seeds and real optimization budget")
    if baselines.snap_k(population) != population:
        raise ValueError("Need an exact archived calibration K")
    candidate_path = Path(candidate_path)
    if not candidate_path.is_file():
        raise ValueError("Missing staged candidate baseline")
    original = json.loads(FROZEN.read_text())
    candidate = json.loads(candidate_path.read_text())
    validation = verify(original, candidate, ["DTLZ1","DTLZ2"])
    assert validation["status"] == "PASS_PROVISIONAL_CLOSENESS_ONLY"
    package_path = (ROOT/"moeabench/diagnostics/resources/references"/
                    name/"calibration_package.npz")
    with np.load(package_path,allow_pickle=False) as pkg:
        gt = np.asarray(pkg["gt_norm"],dtype=float)
        ideal = np.asarray(pkg["ideal"],dtype=float)
        span = np.asarray(pkg["nadir"],dtype=float)-ideal
    if not np.array_equal(gt, np.asarray(original["problems"][name]["gt_reference"])):
        raise ValueError("Frozen baseline and reference GT differ")
    s_k = float(baselines.get_resolution_factor_k(gt, population, seed=0))
    entry = candidate["problems"][name][str(population)]["closeness"]
    if entry.get("ideal_res_method") != "uniform_normalized_objective_pf_area_v1":
        raise ValueError("Candidate lacks the specified normalized-area law")
    eps = float(entry["ideal_res"])
    legacy_blur = np.asarray(
        original["problems"][name][str(population)]["closeness"]["rand_ecdf"],
        dtype=float,
    )
    tree = cKDTree(gt)
    trials = []
    for seed in optimizer_seeds:
        exp = mb.experiment(
            mop=getattr(mb.mops,name)(),
            moea=mb.moeas.NSGA2(population=population,generations=generations)
        )
        exp.run(seed=int(seed), silent=True)
        old = mb.clinic.audit(exp)
        with baselines.use_baselines(str(candidate_path)):
            new = mb.clinic.audit(exp)
        if old.fr is None or new.fr is None or old.quality is None or new.quality is None:
            raise AssertionError("Missing real clinical FAIR report")
        ctx = old.diagnostic_context
        ctxnew = new.diagnostic_context
        if (ctx.get("gt_source") != "registered" or
            not np.array_equal(ctx["GT"],gt) or
            not np.array_equal(ctxnew["GT"],gt) or
            not np.isclose(ctx["s_k"],s_k,rtol=0,atol=1e-12) or
            not np.isclose(ctxnew["s_k"],s_k,rtol=0,atol=1e-12)):
            raise AssertionError("Real audit changed normalized GT or s_K")
        for metric in ("COVERAGE","GAP","REGULARITY","BALANCE","HEADWAY"):
            np.testing.assert_allclose(
                old.fr.metrics[metric].value,new.fr.metrics[metric].value,
                rtol=0,atol=1e-12,equal_nan=True)
            np.testing.assert_allclose(
                old.quality.scores["Q_"+metric].value,
                new.quality.scores["Q_"+metric].value,
                rtol=0,atol=1e-12,equal_nan=True)
        checkpoints = []
        for generation in (0,generations//2,-1):
            physical = np.asarray(exp.front(generation),dtype=float)
            if physical.ndim != 2 or physical.shape[1]!=3 or not len(physical):
                raise ValueError("Missing real NSGA-II non-dominated checkpoint")
            distances = tree.query((physical-ideal)/span,k=1)[0] / s_k
            q_old = clinical_q_for_fixed_front(
                distances,legacy_blur,0.0,k=population)
            q_new = clinical_q_for_fixed_front(
                distances,legacy_blur,eps,k=population)
            clipped = ((q_old==0.0 and q_new==0.0) or
                       (q_old==1.0 and q_new==1.0))
            if generation == -1:
                np.testing.assert_allclose(
                    q_old,old.quality.scores["Q_CLOSENESS"].value,
                    atol=1e-12,rtol=0)
                np.testing.assert_allclose(
                    q_new,new.quality.scores["Q_CLOSENESS"].value,
                    atol=1e-12,rtol=0)
                np.testing.assert_allclose(
                    new.fr.metrics["CLOSENESS"].raw_data,
                    np.maximum(0,np.asarray(old.fr.metrics["CLOSENESS"].raw_data)-eps),
                    atol=1e-12,rtol=0)
            checkpoints.append({
                "generation":generation,
                "front_sha256":_digest(physical),
                "front_points":len(physical),
                "q_frozen":q_old,"q_area_candidate":q_new,
                "absolute_q_shift":abs(q_new-q_old),
                "both_endpoints_clipped":bool(clipped),
                "median_raw_u":float(np.median(distances)),
            })
        trials.append({"optimizer_seed":int(seed),"checkpoints":checkpoints})
    all_rows=[c for t in trials for c in t["checkpoints"]]
    usable=[x for x in all_rows if not x["both_endpoints_clipped"]]
    return {
        "status":"NONCERTIFIED_PAIRED_REAL_MOEA_Q_CANDIDATE",
        "problem":name,"K":population,"generations":generations,
        "optimizer_seeds":list(optimizer_seeds),
        "baseline_frozen_sha256":_sha(FROZEN),
        "baseline_candidate_sha256":_sha(candidate_path),
        "frozen_gt_sha256":_digest(gt),"s_K":s_k,
        "area_ideal_res":eps,"reference_measure":entry["reference_measure"],
        "sample_seed":entry["ideal_res_sampling_seed"],
        "sample_count":entry["ideal_res_sample_count"],
        "all_checkpoint_count":len(all_rows),
        "informative_checkpoint_count":len(usable),
        "clipped_checkpoint_count":len(all_rows)-len(usable),
        "max_informative_absolute_q_shift":(
            max(x["absolute_q_shift"] for x in usable) if usable else None),
        "unrelated_fair_q_unchanged":True,
        "trials":trials,"certified":False,
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--problem",default="DTLZ1",choices=("DTLZ1","DTLZ2"))
    p.add_argument("--optimizer-seed",type=int,action="append",dest="seeds")
    p.add_argument("--population",type=int,default=100)
    p.add_argument("--generations",type=int,default=500)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if (out.exists() or out.is_relative_to(PROTECTED) or
        out.is_relative_to((ROOT/"calibration/reports").resolve()) or
        out==CANONICAL_REPORT):
        p.error("Refusing overwrite or certified report location")
    result=score_fronts(a.problem,a.candidate,
                        optimizer_seeds=tuple(a.seeds or (1,7)),
                        population=a.population,generations=a.generations)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="trials"},indent=2))

if __name__=="__main__":
    main()
