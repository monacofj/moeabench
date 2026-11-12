#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Non-destructive DPF5 real-MOEA Q sensitivity to independent full calibrations.

Research only: issue #81 / proposed ADR 0055. No registered baseline is written.
Each replicate has disjoint ideal, sigma-fit and held-out blur RNG streams.
The same real NSGA-II populations are evaluated against every replicate.
"""
import argparse
import hashlib
import json
import sys
import platform
from importlib.metadata import version
from itertools import product
from pathlib import Path

# Direct invocation by file path sets sys.path[0] to calibration/scripts,
# not the repository root. Mirror the established issue81 validator.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import moeabench as mb
from scipy.spatial import cKDTree

from calibration.scripts.validate_dpf5_closeness_protocol_issue81 import (
    ROOT, FROZEN, RESOURCE_DIR, REPORTS, OFFICIAL, _sha,
    frozen_reference, sample_independent_ideal,
)
from calibration.scripts.generate_baselines import (
    calibrate_sigma, estimate_gt_normals, generate_half_normal_blur, N_SAMPLES,
)
from moeabench.diagnostics import baselines
from moeabench.diagnostics.baselines import get_resolution_factor_k
from moeabench.diagnostics.qscore import compute_q_wasserstein

METRICS = ("COVERAGE", "GAP", "REGULARITY", "BALANCE", "HEADWAY")


def seed_triplet(k, replicate):
    """Unique seed namespaces over a bounded positive integer domain."""
    if not (2 <= k <= 200 and 0 <= replicate < 10000):
        raise ValueError("K or replicate outside designed seed namespace")
    # calibrate_sigma consumes fit_seed + iteration, for iterations 0..5.
    # Reserve the WHOLE controller block, not just its initial seed.
    base = 100000 + 1000000 * k + 10 * replicate
    return base, base + 1, base + 7


def make_replicate(*, model, gt, ideal, nadir, tree, normals, k, scale,
                   replicate, ideal_size):
    ideal_seed, fit_seed, blur_seed = seed_triplet(k, replicate)
    fresh, d_branch = sample_independent_ideal(
        model, size=ideal_size, seed=ideal_seed)
    fresh_norm = (fresh - ideal) / (nadir - ideal)
    ideal_distances = tree.query(fresh_norm)[0]
    if np.min(ideal_distances) <= 1e-10:
        raise AssertionError("Independent ideal sample overlaps frozen GT")
    epsilon = float(np.median(ideal_distances) / scale)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise AssertionError("Independent ideal residual must be positive")
    sigma, fit_median, _ = calibrate_sigma(
        gt, normals, scale, fit_seed, tree=tree)
    if abs(fit_median - 2.0) >= 0.1:
        raise AssertionError("Sigma controller failed its original tolerance")
    blurred = generate_half_normal_blur(
        gt, normals, sigma, np.random.RandomState(blur_seed))
    if not (np.all(blurred >= gt) and np.all(np.any(blurred > gt, axis=1))):
        raise AssertionError("Half-normal reference crossed improving side")
    distances = tree.query(blurred)[0] / scale
    corrected = np.maximum(0.0, distances - epsilon)
    ordered = np.sort(corrected)
    ecdf = ordered[np.linspace(0, len(ordered)-1, N_SAMPLES).astype(int)]
    if not (len(ecdf) == N_SAMPLES and np.all(np.diff(ecdf) >= 0)
            and np.all(np.isfinite(ecdf))):
        raise AssertionError("Invalid reference order statistics")
    return {
        "replicate": replicate, "seeds": {
            "ideal": ideal_seed, "fit": fit_seed, "blur": blur_seed},
        "fit_reserved_seeds": list(range(fit_seed, fit_seed + 6)),
        "ideal_objectives_sha256": _sha(fresh_norm),
        "ideal_D_mass_realized": float(np.mean(d_branch)),
        "epsilon": epsilon, "sigma": float(sigma),
        "fit_normalized_median": float(fit_median),
        "held_out_normalized_median": float(np.median(distances)),
        "blur_objectives_sha256": _sha(blurred),
        "ecdf_sha256": _sha(ecdf),
        "ecdf": ecdf,
    }


def pooled_candidate(candidates):
    """Experimental equal-replicate pooled empirical law, not an approved estimator.

    Each replicate contributes the same number of its stored 200 order
    statistics. This is a mixture of the 200-point compressed ECDFs, NOT
    pooling the original 3367 negative samples (which are not saved here).
    """
    if len(candidates) < 2:
        raise ValueError("Need at least two complete calibration replicates")
    ecdfs = [np.asarray(item["ecdf"], dtype=float) for item in candidates]
    if any(len(a) != N_SAMPLES or not np.all(np.isfinite(a))
           or np.any(np.diff(a) < 0) for a in ecdfs):
        raise ValueError("Invalid candidate ECDF for pooling")
    mixture = np.sort(np.concatenate(ecdfs))
    pooled_ecdf = mixture[
        np.linspace(0, len(mixture) - 1, N_SAMPLES).astype(int)]
    # Research choice: arithmetic mean epsilon before correcting the fixed
    # real distances, while the failure mixture is built from *already*
    # epsilon-corrected reference ECDFs. This is not automatically
    # equivalent to a single full pooled raw-distance calibration.
    epsilon = float(np.mean([item["epsilon"] for item in candidates]))
    return epsilon, pooled_ecdf


def q_from_raw(raw_normalized_distance, epsilon, failure_ecdf):
    """Exact current Q-Closeness W1 mapping on corrected physical distances."""
    u = np.maximum(0.0, np.asarray(raw_normalized_distance) - epsilon)
    return float(compute_q_wasserstein(u, np.zeros_like(u), failure_ecdf))


def _baseline_snapshot(audit):
    return {
        name: (float(audit.fr.metrics[name].value),
               float(audit.quality.scores["Q_" + name].value))
        for name in METRICS
    }


def _ranking(scores):
    # Ties retain a stable index order; report score gaps separately.
    return tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], i)))


def study(*, ks=(40, 100, 200), generations=300, optimizer_seeds=(1, 7),
          replicates=8, ideal_size=16384, algorithms=("NSGA2",), budgets=None):
    if (not ks or len(set(ks)) != len(ks) or
            not optimizer_seeds or len(set(optimizer_seeds)) != len(optimizer_seeds)
            or any(not isinstance(s, int) or s < 0 for s in optimizer_seeds)
            or not 2 <= replicates <= 10000 or ideal_size < 64 or
            ideal_size > 100000 or generations < 2):
        raise ValueError("Invalid experimental design")
    budgets = tuple(budgets or (generations,))
    if (not algorithms or len(set(algorithms)) != len(algorithms)
            or any(a not in ("NSGA2", "SPEA2") for a in algorithms)
            or len(set(budgets)) != len(budgets)
            or any(not isinstance(b, int) or b < 2 for b in budgets)):
        raise ValueError("Invalid algorithms or generation budgets")
    protected = list(RESOURCE_DIR.rglob("*")) + [OFFICIAL]
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in protected if p.is_file()}
    source, gt, ideal, nadir, model = frozen_reference()
    tree = cKDTree(gt)
    normals = estimate_gt_normals(gt)
    by_k = {}
    all_seeds = set()
    for k in ks:
        if not isinstance(k, int) or not 2 <= k <= 200 or (
                str(k) not in source["problems"]["DPF5"]):
            raise ValueError("Unregistered K")
        scale = float(get_resolution_factor_k(gt, k, seed=0))
        candidates = [
            make_replicate(model=model, gt=gt, ideal=ideal, nadir=nadir,
                           tree=tree, normals=normals, k=k, scale=scale,
                           replicate=r, ideal_size=ideal_size)
            for r in range(replicates)
        ]
        for candidate in candidates:
            seeds = (candidate["seeds"]["ideal"], candidate["seeds"]["blur"],
                     *candidate["fit_reserved_seeds"])
            if len(set(seeds)) != 8 or any(s in all_seeds for s in seeds):
                raise AssertionError("Calibration RNG namespaces overlap")
            all_seeds.update(seeds)
        pool_epsilon, pool_ecdf = pooled_candidate(candidates)
        real = []
        for algorithm, budget, seed in product(algorithms, budgets, optimizer_seeds):
            experiment = mb.experiment(
                mop=mb.mops.DPF5(D=2, M=3),
                moea=getattr(mb.moeas, algorithm)(population=k, generations=budget))
            experiment.run(seed=seed, silent=True)
            audit = mb.clinic.audit(experiment)
            if audit.fr is None or audit.quality is None:
                raise AssertionError("Real optimizer audit unavailable")
            ctx = audit.diagnostic_context
            if (ctx.get("gt_source") != "registered"
                    or not np.array_equal(ctx["GT"], gt)
                    or not np.isclose(float(ctx["s_k"]), scale, atol=1e-12, rtol=0)
                    or ctx["P_initial"] is None or ctx["P_final"] is None):
                raise AssertionError("Authentic paired P0/Pf/registered GT required")
            raw_corrected = np.asarray(
                audit.fr.metrics["CLOSENESS"].raw_data, dtype=float)
            old_epsilon = float(source["problems"]["DPF5"][str(k)]
                                ["closeness"].get("ideal_res", 0.0))
            # Better than reconstructing clipped raw data from the old report:
            # recompute true nearest-GT distances from the immutable final points.
            physical = tree.query(np.asarray(ctx["P_final"], dtype=float))[0] / scale
            np.testing.assert_allclose(
                raw_corrected, np.maximum(0.0, physical - old_epsilon),
                atol=1e-10, rtol=0)
            # Independent implementation oracle: recompute the *archived*
            # clinical Q with exactly the existing baseline ECDF and W1 map
            # before any candidate comparisons. This catches normalization,
            # population and baseline-schema mismatches immediately.
            _, _, historical_ecdf = baselines.get_baseline_ecdf(
                "DPF5", baselines.snap_k(k, problem="DPF5"),
                "closeness", target_m=3)
            original_q = q_from_raw(physical, old_epsilon, historical_ecdf)
            np.testing.assert_allclose(
                original_q, float(audit.quality.scores["Q_CLOSENESS"].value),
                atol=1e-10, rtol=0)
            fixed_metrics = _baseline_snapshot(audit)
            population_hashes = (_sha(ctx["P_initial"]), _sha(ctx["P_final"]))
            q_values = [q_from_raw(physical, c["epsilon"], c["ecdf"])
                        for c in candidates]
            pool_q = q_from_raw(physical, pool_epsilon, pool_ecdf)
            if not all(np.isfinite(q) and 0 <= q <= 1
                       for q in (*q_values, pool_q)):
                raise AssertionError("Invalid candidate clinical Q")
            if population_hashes != (_sha(ctx["P_initial"]), _sha(ctx["P_final"])):
                raise AssertionError("Paired population arrays mutated")
            if fixed_metrics != _baseline_snapshot(audit):
                raise AssertionError("Non-Closeness indicators mutated")
            real.append({
                "optimizer": algorithm, "generations": budget, "seed": seed, "K": k,
                "P0_normalized": ctx["P_initial"].tolist(),
                "Pf_normalized": ctx["P_final"].tolist(),
                "raw_normalized_distances": physical.tolist(),
                "evaluated_population_size": len(physical),
                "generation_history_length": len(experiment.moea.F_gen_all),
                "P0_sha256": _sha(ctx["P_initial"]),
                "Pf_sha256": _sha(ctx["P_final"]),
                "historical_q": float(audit.quality.scores["Q_CLOSENESS"].value),
                "candidate_q": q_values,
                "pooled_reference_q_experimental": pool_q,
                "pooled_minus_single_mean_q": pool_q-float(np.mean(q_values)),
                "candidate_q_range": max(q_values)-min(q_values),
                "non_closeness_snapshot": fixed_metrics,
            })
        orders = [_ranking([run["candidate_q"][j] for run in real])
                  for j in range(replicates)]
        by_k[str(k)] = {
            "s_K": scale,
            "experimental_pooled_reference": {
                "epsilon_mean": pool_epsilon, "ecdf": pool_ecdf.tolist(),
                "ecdf_sha256": _sha(pool_ecdf),
                "pooling_method": "equal_mixture_of_200_stored_quantiles",
                "warning": "NOT original 3367 samples; NOT approved",
            },
            "replicates": [{key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in c.items()} for c in candidates],
            "real_runs": real,
            "candidate_ranking_orders": [list(o) for o in orders],
            "ranking_stable": len(set(orders)) == 1,
            "max_real_run_q_range": max(r["candidate_q_range"] for r in real),
        }
    after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in protected if p.is_file()}
    if before != after:
        raise AssertionError("Protected artifacts changed during study")
    return {
        "protected_artifact_sha256": before,
        "normalization_sha256": _sha(np.stack([ideal, nadir])),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / "calibration/scripts/generate_baselines.py")},
        "environment": {"python": platform.python_version(),
                        **{name: version(name) for name in ("numpy", "scipy", "pymoo", "deap", "moeabench")}},
        "seed_design": "v2_reserved_six_fit_streams_stride10",
        "algorithms": list(algorithms), "generation_budgets": list(budgets),
        "status": "NONCERTIFIED_REAL_Q_FULL_CALIBRATION_SENSITIVITY",
        "mop": "DPF5", "M": 3, "D": 2,
        "optimizer": "NSGA2", "generations": generations,
        "optimizer_seeds": list(optimizer_seeds),
        "complete_calibrations_per_K": replicates, "ideal_size": ideal_size,
        "frozen_baseline_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "frozen_GT_sha256": _sha(gt), "source_version": source.get("version"),
        "results_by_K": by_k, "certified": False,
        "no_official_artifacts_modified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, action="append", dest="ks")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--budget", type=int, action="append", dest="budgets")
    parser.add_argument("--algorithm", choices=("NSGA2", "SPEA2"), action="append", dest="algorithms")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--ideal-size", type=int, default=16384)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dest = args.output.resolve()
    if (dest.exists() or dest.is_relative_to(RESOURCE_DIR)
            or dest.is_relative_to(REPORTS) or dest == OFFICIAL):
        parser.error("Refusing existing or protected output path")
    result = study(ks=tuple(args.ks or (40, 100, 200)),
                   generations=args.generations,
                   optimizer_seeds=tuple(args.seeds or (1, 7)),
                   replicates=args.replicates, ideal_size=args.ideal_size,
                   algorithms=tuple(args.algorithms or ("NSGA2",)), budgets=args.budgets)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    for k, item in result["results_by_K"].items():
        print("K", k, "max real-Q range", item["max_real_run_q_range"],
              "ranking stable", item["ranking_stable"])
    print("NONCERTIFIED. No official baseline, GT or audit report modified.")


if __name__ == "__main__":
    main()
