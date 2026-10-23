#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #75: challenge PF-generator ideal_res with independent geometric laws.

DTLZ1 M=3: uniform surface-area measure on the 0.5-simplex (Dirichlet).
DTLZ2 M=3: uniform surface-area measure on the positive unit sphere
             (absolute, normalized isotropic Gaussian directions).
These laws are generated WITHOUT calling MoeaBench's ps()/pf() routines.
DPF5: exploratory random Pareto-*set* decision samples followed by the
current MOP objective evaluator. This is independent of the ordered/grid
PF SAMPLER, not an independent proof of Pareto optimality, PF geometry or
surface-area uniformity. The distinction is explicit in the output.

Observe the exact SAME frozen GT, original ECDF and actual NSGA-II fronts.
Only the candidate ideal_res varies. Reuse actual production Q_CLOSENESS
scoring and fail closed if the frozen final-front control disagrees with
mb.clinic.audit(exp). Never write a certified baseline/audit report.
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
import moeabench as mb
from moeabench.diagnostics import baselines
from moeabench.diagnostics.calibration import independent_analytical_reference
from calibration.scripts.study_ideal_residue_robustness import _stats, _region_hist
from calibration.scripts.study_real_moea_q_sensitivity import _digest, _moea_trial

ELIGIBLE = {"DTLZ1", "DTLZ2", "DPF5"}
RESTRICTED = tuple((ROOT / p).resolve() for p in
                   ("moeabench/diagnostics/resources", "calibration/reports"))


def independent_physical_front(name, count, seed, *, mop=None):
    """Use a second mathematical construction, not the internal PF sampler."""
    if name not in ELIGIBLE or count < 2:
        raise ValueError("Unsupported MOP or insufficient independent points")
    rng = np.random.default_rng(seed)
    if name == "DTLZ1":
        # Dirichlet(1,...,1) has constant density in barycentric coordinates,
        # hence uniform (Euclidean area) distribution on an affine simplex.
        return 0.5 * rng.dirichlet(np.ones(3), size=count)
    if name == "DTLZ2":
        directions = np.abs(rng.standard_normal((count, 3)))
        norm = np.linalg.norm(directions, axis=1)
        if np.any(norm <= 0):
            raise ValueError("Zero-length Gaussian direction")
        return directions / norm[:, None]

    # DPF5 PS constraint (its implementation): x_M = x_1 and trailing x_i=.5.
    # Sample x_1 and x_2 independently instead of traversing the built-in
    # rectangular decision grid. This uses the same evaluator and is NOT
    # an independently established uniform measure on the objective PF.
    if mop is None:
        mop = mb.mops.DPF5()
    x = np.full((count, mop.N), 0.5, dtype=float)
    x[:, :mop.M-1] = rng.random((count, mop.M-1))
    x[:, mop.M-1] = x[:, 0]
    values = np.asarray(mop.evaluation(x)["F"], dtype=float)
    if values.shape != (count, mop.M) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid DPF5 evaluation of independently sampled PS")
    return values


def _normalize_unique_fresh(physical, gt, ideal, span, *, count):
    norm = (np.asarray(physical, dtype=float) - ideal) / span
    if norm.ndim != 2 or norm.shape[1] != gt.shape[1] or not np.all(np.isfinite(norm)):
        raise ValueError("Bad analytical objective dimensions or values")
    tree = cKDTree(gt)
    threshold = max(1e-12, 1e-10*float(np.max(np.ptp(gt, axis=0))))
    # Exact duplicate objective vectors are redundant evidence, not independent
    # samples, even if their decision-space origins are different.
    points = np.unique(norm, axis=0)
    distances = tree.query(points, k=1)[0]
    points = points[distances > threshold]
    if len(points) < count:
        raise ValueError("Independent reference lacks enough unique non-GT samples")
    return points[:count]


def _dist_stats(gt, sample):
    nearest = cKDTree(gt).query(sample, k=1)[0]
    reverse = cKDTree(sample).query(gt, k=1)[0]
    local = float(np.median(cKDTree(gt).query(gt, k=2)[0][:, 1]))
    if local <= 0:
        raise ValueError("Archived GT has no finite local spacing")
    return {
        "candidate_to_frozen_GT_over_local_NN": {
            "median": float(np.median(nearest)/local),
            "p95": float(np.percentile(nearest, 95)/local),
        },
        "frozen_GT_to_candidate_over_local_NN": {
            "median": float(np.median(reverse)/local),
            "p95": float(np.percentile(reverse, 95)/local),
        },
        "f1_region_counts": _region_hist(sample, gt),
    }


def candidate_references(name, gt, ideal, nadir, *,
                         seeds=(123,124,125,126), sample_size=1024):
    """Make both reference laws from *different* generators and RNGs."""
    if not seeds or len(seeds) != len(set(seeds)) or sample_size < 2:
        raise ValueError("Need distinct seeds and a positive sample size")
    gt = np.asarray(gt, dtype=float)
    ideal, nadir = np.asarray(ideal, dtype=float), np.asarray(nadir, dtype=float)
    span = nadir-ideal
    if np.any(span <= 0) or gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError("Require calibrated M=3 GT with strictly positive spans")
    mop = getattr(mb.mops, name)()

    class CalibratedInternal:
        def pf(self, n_points):
            return (np.asarray(mop.pf(n_points=n_points),dtype=float)-ideal)/span

    # Demand that our geometrical analytical construction agrees with the
    # known physical PF manifold, not merely that Q scores happen to be close.
    out = {}
    for seed in seeds:
        physical = independent_physical_front(name, sample_size * 2, seed, mop=mop)
        if name == "DTLZ1" and not np.allclose(physical.sum(axis=1), 0.5, atol=1e-12):
            raise AssertionError("Independent DTLZ1 sample does not lie on simplex")
        if name == "DTLZ2" and not np.allclose(
            np.linalg.norm(physical,axis=1), 1.0, atol=1e-12
        ):
            raise AssertionError("Independent DTLZ2 sample does not lie on sphere")
        independent = _normalize_unique_fresh(
            physical, gt, ideal, span, count=sample_size
        )
        internal = independent_analytical_reference(
            CalibratedInternal(), gt, n_fresh=sample_size, sampling_seed=seed,
        )
        if internal is None or len(internal) != sample_size:
            raise ValueError("Internal PF cannot produce enough fresh ideal points")
        for method, sample in (("geometric" if name != "DPF5" else "decision_ps",
                                independent), ("internal_pf", internal)):
            key = f"{method}:{seed}"
            dist = cKDTree(gt).query(sample,k=1)[0]
            if np.any(dist <= 1e-12):
                raise ValueError(f"{key}: candidate overlaps frozen GT")
            out[key] = {
                "eps_distance_median": float(np.median(dist)),
                "sha256": _digest(sample),
                "regions_f1": _region_hist(sample,gt),
                "sample": sample,
                "geometry": _dist_stats(gt,sample),
            }
    return out


def evaluate(name, *, optimizer_seeds=(1,7), reference_seeds=(123,124,125,126),
             population=None, generations=None, sample_size=1024):
    if name not in ELIGIBLE:
        raise ValueError(f"{name}: no independent geometry oracle")
    # DTLZ1 needs a converged front: prior pop40/gen40 gave Q=0 everywhere.
    if population is None:
        population = 100 if name == "DTLZ1" else 40
    if generations is None:
        generations = 500 if name == "DTLZ1" else 40
    if len(optimizer_seeds) < 2 or len(reference_seeds) < 2:
        raise ValueError("At least two optimizer and reference seeds required")
    with (ROOT/"moeabench/diagnostics/resources/baselines_v0.14.0.json").open(
        encoding="utf-8"
    ) as fh:
        frozen = json.load(fh)
    with np.load(
        ROOT/"moeabench/diagnostics/resources/references"/name/"calibration_package.npz",
        allow_pickle=False,
    ) as pkg:
        gt = np.asarray(pkg["gt_norm"],dtype=float)
        ideal = np.asarray(pkg["ideal"],dtype=float)
        nadir = np.asarray(pkg["nadir"],dtype=float)
    record = frozen["problems"][name]
    if not np.array_equal(gt,np.asarray(record["gt_reference"],dtype=float)):
        raise ValueError("Archived GT differs from packaged GT")
    k = population
    if baselines.snap_k(k) != k:
        raise ValueError(f"Population K={k} has no exact archived calibration")
    s_k = baselines.get_resolution_factor_k(gt,k,seed=0)
    null = {int(kk):np.asarray(v["closeness"]["rand_ecdf"],dtype=float)
            for kk,v in record.items() if kk.isdigit() and "closeness" in v}
    if k not in null:
        raise ValueError("Missing K-matched Closeness ECDF")
    samples = candidate_references(
        name, gt, ideal, nadir, seeds=reference_seeds, sample_size=sample_size
    )
    eps = {key:float(val["eps_distance_median"]/s_k) for key,val in samples.items()}
    trials = [_moea_trial(
        name,gt,s_k,null,seed=seed,population=population,
        generations=generations,normalized_ideal=ideal,
        normalized_span=nadir-ideal,analytical_samples=samples,
    ) for seed in optimizer_seeds]
    method = "decision_ps" if name=="DPF5" else "geometric"
    comparisons = []
    for trial in trials:
        for checkpoint, front in trial["fronts"].items():
            a = [front["q_by_analytical_seed"][f"{method}:{seed}"]
                 for seed in reference_seeds]
            b = [front["q_by_analytical_seed"][f"internal_pf:{seed}"]
                 for seed in reference_seeds]
            comparisons.append({
                "run_seed":trial["seed"],"checkpoint":checkpoint,
                "front_sha256":front["front_sha256"],
                "Q_old":front["archived_zero_residue_q"],
                "Q_external":_stats(a), "Q_internal":_stats(b),
                "absolute_median_gap":float(abs(np.median(a)-np.median(b))),
                "Q_clipped_zero_both":bool(max(a)==0 and max(b)==0),
                "Q_clipped_one_both":bool(min(a)==1 and min(b)==1),
            })
    return {
        "status":"NONCERTIFIED_CROSS_GENERATOR_PF_Q_CHALLENGE",
        "mop":name,"gt_sha256":_digest(gt),
        "reference_method":method,
        "independence_scope":(
            "External closed-form uniform surface sampling, independent of "
            "MoeaBench pf() (M=3 simplex/sphere)" if name!="DPF5" else
            "Monte Carlo decision-set construction independent of pf() grid; "
            "NOT independently proved uniform or Pareto-complete PF"
        ),
        "optimization":{"seeds":list(optimizer_seeds),"population":population,
                        "generations":generations},
        "reference_seeds":list(reference_seeds),"sample_size":sample_size,
        "s_K":float(s_k),
        "calibration":{key:{
            "epsilon":eps[key],"sha256":v["sha256"],
            "geometry":v["geometry"],
        } for key,v in samples.items()},
        "epsilon_distribution":{
            kind:_stats([eps[f"{kind}:{s}"] for s in reference_seeds])
            for kind in (method,"internal_pf")
        },
        "real_front_comparison":comparisons,
        "max_between_method_Q_gap":max(x["absolute_median_gap"] for x in comparisons),
        "nonclipped_comparisons":sum(
            not (x["Q_clipped_zero_both"] or x["Q_clipped_one_both"])
            for x in comparisons
        ),
        "official_baselines_changed":False,
        "caveat":"This experiment does not establish a scientifically acceptable "
                 "quality-score drift threshold or a new certified calibration.",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem",action="append",dest="problems",required=True)
    p.add_argument("--output",type=Path,required=True)
    opts=p.parse_args()
    output=opts.output.resolve()
    if (output.exists() or any(output.is_relative_to(path) for path in RESTRICTED)):
        p.error("Refusing overwrite or placement in certified directories")
    results={"status":"NONCERTIFIED_CROSS_GENERATOR_PF_Q_CHALLENGE",
             "results":{},"failures":{}}
    for name in opts.problems:
        try:
            rec=evaluate(name)
            results["results"][name]=rec
            print(json.dumps({
                "problem":name,"mode":rec["reference_method"],
                "eps":rec["epsilon_distribution"],
                "Q_cross_method_max":rec["max_between_method_Q_gap"],
                "usable_nonclipped_fronts":rec["nonclipped_comparisons"],
                "all_fronts":len(rec["real_front_comparison"]),
            }),flush=True)
        except (ValueError,AssertionError,KeyError,TypeError,OSError) as exc:
            results["failures"][name]=f"{type(exc).__name__}: {exc}"
            print("UNSUPPORTED",name,results["failures"][name],flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    if results["failures"]:
        raise SystemExit("Independent PF challenge failed; see preserved JSON")


if __name__=="__main__":
    main()
