#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Issue #75: Q_CLOSENESS residue sensitivity on genuine MOEA fronts.

For each MOP and execution seed, actually optimize the MOP with NSGA-II.
Freeze the non-dominated objective fronts at the initial, intermediate and
final generations. Evaluate each *unchanged* front under multiple
independently selected analytical PF subsets; only finite-GT ideal_res and
its corresponding corrected blur ECDF vary.

Clinical score computation calls the production Q_CLOSENESS mapping.
Audited frozen-baseline final-front data and exact GT normalization are
checked independently before reporting. This cannot certify a sampling
law, a new Q-score baseline, or results for unsupported DPF3/DTLZ8.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree
import moeabench as mb

from moeabench.diagnostics import baselines, qscore
from moeabench.diagnostics.calibration import (
    estimate_ideal_residue, independent_analytical_reference,
)
from calibration.scripts.study_ideal_residue_robustness import _stats, _region_hist

CERTIFIED = (
    (ROOT / "moeabench/diagnostics/resources").resolve(),
    (ROOT / "calibration/reports").resolve(),
)
MOPS = ("DTLZ1", "DTLZ2", "DPF5")


def _digest(matrix):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(matrix, dtype=np.float64)
    ).tobytes()).hexdigest()


def clinical_q_for_fixed_front(raw_distances, frozen_blur, epsilon, *, k):
    """Evaluate the *real* Q-score on paired, consistently corrected data.

    We inject only the reference ECDF into the scoring function (not into
    installed baselines) so each epsilon hypothesis reuses exactly the same
    MOEA front and the same frozen blur measurements. No heuristic surrogate
    Q formula is used.
    """
    u = np.asarray(raw_distances, dtype=float)
    blur = np.asarray(frozen_blur, dtype=float)
    epsilon = float(epsilon)
    if (u.ndim != 1 or len(u) == 0 or blur.ndim != 1 or len(blur) != 200 or
            np.any(~np.isfinite(u)) or np.any(~np.isfinite(blur)) or
            np.any(u < 0) or np.any(blur < 0) or
            not np.isfinite(epsilon) or epsilon < 0):
        raise ValueError("Expected finite nonnegative front/reference and epsilon")
    corrected_u = np.maximum(0.0, u-epsilon)
    corrected_null = np.maximum(0.0, blur-epsilon)
    with patch.object(
        baselines, "snap_k", side_effect=lambda K, problem=None: K
    ), patch.object(
        baselines, "get_baseline_ecdf",
        return_value=(0.0, float(np.median(corrected_null)), corrected_null)
    ):
        result = float(qscore.q_closeness(
            corrected_u, problem="RealMOEAResidueSensitivity", k=k,
        ).value)
    if not np.isfinite(result) or result < 0 or result > 1:
        raise AssertionError("Invalid real implementation Q_CLOSENESS")
    return result


def _analytical_samples(name, gt, package, *, n_fresh, seeds):
    """Verify frozen GT and produce independent-of-GT unique PF subsets."""
    from calibration.scripts.generate_baselines import _analytical_ideal_from_package
    if _analytical_ideal_from_package(name, gt, package) is None:
        raise ValueError(f"{name}: analytical PF incompatible with frozen GT")
    ideal = np.asarray(package["ideal"], dtype=float)
    span = np.asarray(package["nadir"], dtype=float)-ideal
    if np.any(span <= 0):
        raise ValueError("Invalid archived calibration normalization")

    class CalibratedPF:
        def __init__(self, native):
            self.native = native
        def pf(self, n_points):
            return (np.asarray(self.native.pf(n_points=n_points))-ideal)/span

    native = getattr(mb.mops, name)()
    calibrated = CalibratedPF(native)
    samples = {}
    for seed in seeds:
        sample = independent_analytical_reference(
            calibrated, gt, n_fresh=n_fresh, sampling_seed=int(seed),
        )
        if sample is None or len(sample) != n_fresh:
            raise ValueError(f"{name}: missing {n_fresh} unique analytical points")
        dist = cKDTree(gt).query(sample, k=1)[0]
        if np.any(dist <= 1e-12):
            raise ValueError("Analytical sample overlaps GT")
        samples[str(seed)] = {
            "eps_distance_median": float(np.median(dist)),
            "sha256": _digest(sample),
            "regions_f1": _region_hist(sample, gt),
            "sample": sample,
        }
    if len(set(s["sha256"] for s in samples.values())) < 2:
        raise ValueError("Analytical seeds yielded identical candidate samples")
    return samples, ideal, span


def _moea_trial(name, gt, s_k, archive, *, seed, population, generations,
                normalized_ideal, normalized_span, analytical_samples):
    """Run a real NSGA-II and return Q changes for each *frozen* front."""
    exp = mb.experiment(
        mop=getattr(mb.mops, name)(),
        moea=mb.moeas.NSGA2(population=population, generations=generations),
    )
    exp.run(seed=int(seed), silent=True)
    audited = mb.clinic.audit(exp)
    if audited.fr is None or audited.quality is None:
        raise ValueError(f"{name}: registered clinical baseline unavailable")
    ctx = audited.diagnostic_context
    if (ctx.get("gt_source") != "registered" or
            not np.array_equal(np.asarray(ctx["GT"]), gt) or
            ctx.get("normalization") is None):
        raise ValueError(f"{name}: actual clinical audit used incompatible GT")
    if not np.isclose(ctx["s_k"], s_k, atol=1e-12, rtol=0):
        raise ValueError("Physical and clinical FPS scales disagree")
    if int(ctx["k"]) not in archive:
        raise ValueError(f"Missing K={ctx['k']} archived ECDF")
    clinical_norm = ctx["normalization"]
    if (not np.allclose(clinical_norm["ideal"], normalized_ideal, atol=0, rtol=0)
            or not np.allclose(
                np.asarray(clinical_norm["nadir"])-np.asarray(clinical_norm["ideal"]),
                normalized_span, atol=0, rtol=0)):
        raise ValueError("Population normalization differs from frozen ideal/nadir")

    k = int(ctx["k"])
    blur = archive[k]
    steps = (0, generations//2, -1)
    output = {"seed":seed, "K":k, "fronts":{}}
    for generation in steps:
        raw_front = np.asarray(exp.front(generation), dtype=float)
        if raw_front.ndim != 2 or raw_front.shape[1] != gt.shape[1] or len(raw_front) == 0:
            raise ValueError(f"{name}: missing non-dominated front at generation {generation}")
        p = (raw_front-normalized_ideal)/normalized_span
        distances = cKDTree(gt).query(p, k=1)[0]
        raw_u = distances/s_k
        baseline_q = clinical_q_for_fixed_front(raw_u, blur, 0.0, k=k)
        if generation == -1:
            reported = float(audited.quality.scores["Q_CLOSENESS"].value)
            if not np.isclose(baseline_q, reported, rtol=0, atol=1e-12):
                raise AssertionError(
                    f"{name}: fixed-front Q disagrees with actual audit: {baseline_q} vs {reported}"
                )
            np.testing.assert_allclose(
                raw_u, np.asarray(audited.fr.metrics["CLOSENESS"].raw_data),
                rtol=0, atol=1e-10,
            )
        q_scores = {}
        for analytical_seed, sample in analytical_samples.items():
            eps = sample["eps_distance_median"]/s_k
            q_scores[analytical_seed] = clinical_q_for_fixed_front(
                raw_u, blur, eps, k=k
            )
        qs = list(q_scores.values())
        low, high = min(qs), max(qs)
        output["fronts"][str(generation)] = {
            "front_sha256": _digest(raw_front),
            "front_points": len(raw_front),
            "raw_closeness_median": float(np.median(raw_u)),
            "archived_zero_residue_q": baseline_q,
            "q_by_analytical_seed": q_scores,
            "q_range": _stats(qs),
            "max_absolute_q_difference": float(high-low),
            "absolute_drift_from_archived": {
                "min": float(min(abs(q-baseline_q) for q in qs)),
                "max": float(max(abs(q-baseline_q) for q in qs)),
            },
            "all_q_clipped_zero": bool(high == 0.0),
            "all_q_clipped_one": bool(low == 1.0),
        }
    return output


def analyze(name, *, run_seeds=(7,8), analytical_seeds=(123,124,125,126),
            population=40, generations=40, n_fresh=1024):
    if name not in MOPS:
        raise ValueError(f"Not an approved analytical pilot MOP: {name}")
    if (generations < 4 or population < 12 or population%4 or
            len(run_seeds)<2 or len(analytical_seeds)<2 or
            len(set(run_seeds))!=len(run_seeds) or
            len(set(analytical_seeds))!=len(analytical_seeds)):
        raise ValueError("Require >=2 independent optimizer and analytical seeds")
    with (ROOT/"moeabench/diagnostics/resources/baselines_v0.14.0.json").open() as f:
        frozen=json.load(f)
    reference_path = ROOT/"moeabench/diagnostics/resources/references"/name/"calibration_package.npz"
    with np.load(reference_path, allow_pickle=False) as pkg:
        gt=np.asarray(pkg["gt_norm"],dtype=float)
        if not np.array_equal(gt, np.asarray(frozen["problems"][name]["gt_reference"])):
            raise ValueError("Archived baseline GT differs from calibration package")
        samples, ideal, span = _analytical_samples(
            name, gt, pkg, n_fresh=n_fresh, seeds=analytical_seeds,
        )
    k=population   # 3-objective NSGA2 at population<=50 uses exact K
    if baselines.snap_k(k) != k:
        raise ValueError("Population does not map to same archived K")
    scale=float(baselines.get_resolution_factor_k(gt,k,seed=0))
    reference={int(key): np.asarray(val["closeness"]["rand_ecdf"], dtype=float)
               for key,val in frozen["problems"][name].items()
               if key.isdigit() and "closeness" in val}
    if k not in reference:
        raise ValueError(f"Missing K={k} frozen reference")
    metadata = {seed:{
        "sha256":rec["sha256"],
        "regions_f1":rec["regions_f1"],
        "ideal_res":rec["eps_distance_median"]/scale,
    } for seed,rec in samples.items()}
    record={
        "status":"NONCERTIFIED_REAL_MOEA_Q_RESIDUE_SENSITIVITY",
        "mop":name, "algorithm":"NSGA2", "optimizer_seeds":list(run_seeds),
        "analytical_seeds":list(analytical_seeds),
        "population":population, "generations":generations,
        "GT_sha256":_digest(gt), "GT_points":len(gt),
        "K":k, "s_K":scale, "frozen_ECDF_sha256":_digest(reference[k]),
        "calibration":metadata, "trials":[],
        "limitations":"NSGA2 real fronts; analytical samples share one deterministic PF law. "
                      "Results include clipping and do not establish arbitrary-front Q stability.",
    }
    for seed in run_seeds:
        record["trials"].append(_moea_trial(
            name,gt,scale,reference,
            seed=int(seed),population=population,generations=generations,
            normalized_ideal=ideal,normalized_span=span,
            analytical_samples=samples,
        ))
    all_fronts=[entry for trial in record["trials"]
                for entry in trial["fronts"].values()]
    record["front_count"]=len(all_fronts)
    record["max_q_span"]=float(max(x["max_absolute_q_difference"] for x in all_fronts))
    record["max_drift_from_frozen_q"]=float(max(
        x["absolute_drift_from_archived"]["max"] for x in all_fronts
    ))
    record["clipped_zero_fronts"]=sum(x["all_q_clipped_zero"] for x in all_fronts)
    record["clipped_one_fronts"]=sum(x["all_q_clipped_one"] for x in all_fronts)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem",action="append",dest="problems",required=True)
    p.add_argument("--run-seed",type=int,action="append",dest="run_seeds")
    p.add_argument("--analytical-seed",type=int,action="append",dest="analytical_seeds")
    p.add_argument("--population",type=int,default=40)
    p.add_argument("--generations",type=int,default=40)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or any(out.is_relative_to(path) for path in CERTIFIED):
        p.error("Refusing to overwrite or publish among certified FAIR baselines/reports")
    result={"status":"NONCERTIFIED_REAL_MOEA_Q_RESIDUE_SENSITIVITY",
            "results":{}, "failures":{}}
    for name in a.problems:
        try:
            rec=analyze(name,run_seeds=tuple(a.run_seeds or (7,8)),
                        analytical_seeds=tuple(a.analytical_seeds or (123,124,125,126)),
                        population=a.population,generations=a.generations)
            result["results"][name]=rec
            print(json.dumps({
                "mop":name, "front_count":rec["front_count"],
                "max_q_span":rec["max_q_span"],
                "max_drift_from_frozen_q":rec["max_drift_from_frozen_q"],
                "clipped_zero_fronts":rec["clipped_zero_fronts"],
                "clipped_one_fronts":rec["clipped_one_fronts"],
                "analytical_epsilons":{
                    k:v["ideal_res"] for k,v in rec["calibration"].items()
                },
            }),flush=True)
        except (ValueError, AssertionError, OSError, KeyError, TypeError) as exc:
            result["failures"][name]=f"{type(exc).__name__}: {exc}"
            print("FAILED",name,result["failures"][name],flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    if result["failures"]:
        raise SystemExit("Real MOEA Q-sensitivity pilot has failed: see artifact")


if __name__=="__main__":
    main()
