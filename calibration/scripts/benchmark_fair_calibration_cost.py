#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bounded FAIR calibration cost study on the frozen GT packages (issue #75).

Unlike the legacy full baseline generator, this intentionally excludes
HEADWAY. A single random/ideal population cannot calibrate the longitudinal
GD95(P_final)/GD95(P_initial) ratio. A genuine Headway calibration needs
paired trajectories, which are not available in these sampling loops.

Samples use the same frozen GT, FPS/K, clusters and physical FAIR primitives
as the v4 generator. Runs at their full 200 random / 30 ideal draw counts
for one K at a time to measure practical per-K costs. The output is only
a NONCERTIFIED timing/validation report, not a production baseline.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist

from moeabench.diagnostics import baselines, fair
from moeabench.diagnostics.calibration import estimate_ideal_residue
from calibration.scripts.generate_baselines import (
    _analytical_ideal_from_package, estimate_gt_normals,
    calibrate_sigma, generate_half_normal_blur,
)

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("COVERAGE", "GAP", "REGULARITY", "BALANCE")


def _stats(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Nonfinite or missing calibration samples")
    return {"median": float(np.median(x)),
            "p05": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95))}


def _measure(pop, gt, u_ref, centroids, ref_hist):
    """Only four physical one-population metrics; no pseudo-Headway."""
    result = {
        "COVERAGE": float(fair.coverage(pop, ref=gt).value),
        "GAP": float(fair.gap(pop, ref=gt).value),
        "REGULARITY": float(fair.regularity(pop, ref_distribution=u_ref).value),
        "BALANCE": float(fair.balance(
            pop, centroids=centroids, ref_hist=ref_hist,
        ).value),
    }
    if set(result) != set(METRICS) or not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Invalid physical calibration measurement")
    return result


def benchmark_problem(problem, *, k=20, n_random=200, n_ideal=30):
    if k < 2 or n_random < 1 or n_ideal < 1:
        raise ValueError("Invalid pilot sampling parameters")
    package_path = (ROOT / "moeabench/diagnostics/resources/references"
                    / problem / "calibration_package.npz")
    if not package_path.is_file():
        raise FileNotFoundError(package_path)
    started = time.perf_counter()
    with np.load(package_path, allow_pickle=False) as package:
        gt = np.asarray(package["gt_norm"], dtype=float)
        independent = _analytical_ideal_from_package(problem, gt, package)
    if gt.ndim != 2 or len(gt) < k or not np.all(np.isfinite(gt)):
        raise ValueError("Invalid GT")
    time_load = time.perf_counter() - started

    prep = time.perf_counter()
    normals = estimate_gt_normals(gt)
    centroids, _ = baselines.get_ref_clusters(gt, c=32, seed=0)
    u_ref = baselines.get_ref_uk(gt, k, seed=0)
    s_k = baselines.get_resolution_factor_k(gt, k, seed=0)
    assignments = np.argmin(cdist(u_ref, centroids), axis=1)
    ref_hist = np.bincount(assignments, minlength=len(centroids)).astype(float)
    ref_hist /= ref_hist.sum()
    time_prepare = time.perf_counter() - prep

    blur_start = time.perf_counter()
    from scipy.spatial import cKDTree
    tree = cKDTree(gt)
    sigma, _, _ = calibrate_sigma(gt, normals, s_k, 1000 + k, tree=tree)
    rng_blur = np.random.RandomState(1500 + k)
    g_blur = generate_half_normal_blur(gt, normals, sigma, rng_blur)
    u_blur = tree.query(g_blur, k=1)[0] / s_k
    if independent is not None:
        epsilon = estimate_ideal_residue(gt, independent, s_k)
        residue_source = "independent_analytical_pf"
    else:
        epsilon = 0.0
        residue_source = "unavailable_no_independent_pf"
    corrected_blur = np.maximum(0.0, u_blur - epsilon)
    time_blur = time.perf_counter() - blur_start

    random_start = time.perf_counter()
    draws_random = {metric: [] for metric in METRICS}
    rng = np.random.RandomState(1000 + k)
    for _ in range(n_random):
        pop = rng.rand(k, gt.shape[1])
        metrics = _measure(pop, gt, u_ref, centroids, ref_hist)
        for metric, value in metrics.items():
            draws_random[metric].append(value)
    time_random = time.perf_counter() - random_start

    ideal_start = time.perf_counter()
    draws_ideal = {metric: [] for metric in METRICS}
    for j in range(n_ideal):
        pop = baselines.get_ref_uk(gt, k, seed=100 + j)
        metrics = _measure(pop, gt, u_ref, centroids, ref_hist)
        for metric, value in metrics.items():
            draws_ideal[metric].append(value)
    time_ideal = time.perf_counter() - ideal_start

    times = {
        "load_and_ideal_source_s": time_load,
        "normal_FPS_clusters_s": time_prepare,
        "blur_and_residue_s": time_blur,
        "random_samples_s": time_random,
        "ideal_samples_s": time_ideal,
    }
    times["total_wall_s"] = sum(times.values())
    return {
        "problem": problem,
        "gt_shape": list(gt.shape),
        "gt_sha256": hashlib.sha256(np.ascontiguousarray(gt).tobytes()).hexdigest(),
        "k": k,
        "s_k": float(s_k),
        "random_draws": n_random,
        "ideal_draws": n_ideal,
        "residue": float(epsilon),
        "residue_source": residue_source,
        "closeness_blur": _stats(corrected_blur),
        "metrics_random": {key:_stats(val) for key,val in draws_random.items()},
        "metrics_ideal": {key:_stats(val) for key,val in draws_ideal.items()},
        "headway": None,
        "headway_status": "NOT_CALIBRATED_requires_true_P0_Pfinal_pairs",
        "timing_seconds": times,
    }


def run_benchmark(problems, *, k=20, n_random=200, n_ideal=30):
    reports = []
    errors = {}
    for name in problems:
        try:
            result = benchmark_problem(name, k=k, n_random=n_random, n_ideal=n_ideal)
            reports.append(result)
            print(json.dumps({"problem":name, "time_seconds":result["timing_seconds"],
                              "ideal_res":result["residue"],
                              "residue_source":result["residue_source"]}), flush=True)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            print(json.dumps({"problem":name,"failure":errors[name]}),flush=True)
    return {
        "status": "NONCERTIFIED_COST_PILOT",
        "cannot_replace_official_baselines": True,
        "headway_omitted": True,
        "k_values": [k],
        "problem_results": reports,
        "failures": errors,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem",action="append",dest="problems",required=True)
    p.add_argument("--k",type=int,default=20)
    p.add_argument("--random-draws",type=int,default=200)
    p.add_argument("--ideal-draws",type=int,default=30)
    p.add_argument("--output",type=Path,required=True)
    opts=p.parse_args()
    target=opts.output.resolve()
    forbidden=(ROOT/"moeabench/diagnostics/resources").resolve()
    if target.is_relative_to(forbidden) or target.is_relative_to((ROOT/"calibration/reports").resolve()):
        p.error("Cost pilot cannot be written to certified calibration/report files")
    if target.exists():
        p.error("Refusing to overwrite existing output")
    results=run_benchmark(opts.problems,k=opts.k,n_random=opts.random_draws,
                          n_ideal=opts.ideal_draws)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    if results["failures"]:
        raise SystemExit("At least one calibration cost pilot failed; see JSON output")


if __name__=="__main__":
    main()
