#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""FAIR #75: scientific robustness challenge for independent ideal_res sampling.

Do NOT treat repetitions from one deterministic analytical PF as independent
new calls: DTLZ1/2 restart their RNG, and DPF5 traverses an ordered grid.
Construct a much larger fresh analytical pool, then use multiple *disjoint*
random subsets for each fixed sample size. Compare the legacy 1024-point
prefix to representative subsets and measure how much this changes epsilon
and Q_CLOSENESS for the same fixed observation distribution.

This is a NONCERTIFIED sensitivity study, not a baseline generator.
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
from unittest.mock import patch

from moeabench import mops
from moeabench.diagnostics import baselines, qscore
from moeabench.diagnostics.calibration import (
    estimate_ideal_residue, independent_analytical_reference,
)

PROBLEMS = ("DTLZ1", "DTLZ2", "DPF5")


def _stats(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("No finite values for sensitivity summary")
    return {
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "median": float(np.median(v)),
        "relative_range_over_median": float((np.max(v) - np.min(v)) /
                                             max(1e-12, abs(np.median(v)))),
    }


def _region_hist(points, gt, *, bins=5):
    """Quantile-spaced regions of GT objective zero; return occupancy counts.

    This is an interpretable one-axis coverage diagnostic, NOT a proof that
    coverage on a nonlinear high-dimensional manifold is representative.
    """
    gt = np.asarray(gt, dtype=float)
    points = np.asarray(points, dtype=float)
    quantiles = np.quantile(gt[:, 0], np.linspace(0, 1, bins+1))
    # Bin by nearest archived point, not candidate's raw f1 alone.
    indices = cKDTree(gt).query(points, k=1)[1]
    objective = gt[indices, 0]
    bucket = np.searchsorted(quantiles[1:-1], objective, side="right")
    return np.bincount(bucket, minlength=bins).astype(int).tolist()


def _fixed_q(raw_u, raw_blur, epsilon, *, k=100):
    """Run the real Q-Closeness implementation with a *staged* ECDF.

    Raw observations are frozen independently of epsilon. Apply the physical
    correction to both observation and reference, then call qscore.q_closeness
    without touching the registered/official clinical baselines.
    """
    observed = np.maximum(0, np.asarray(raw_u) - epsilon)
    null = np.maximum(0, np.asarray(raw_blur) - epsilon)
    with patch.object(baselines, "snap_k", side_effect=lambda K, problem=None: K), \
         patch.object(baselines, "get_baseline_ecdf",
                      return_value=(0.0, float(np.median(null)), null)):
        value = float(qscore.q_closeness(
            observed, problem="SensitivityPilot", k=k,
        ).value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise AssertionError("Invalid clinical quality calculation")
    return value


def compare_subsets(gt, fresh_pool, old_prefix, raw_blur, *,
                    sample_sizes=(256, 512, 1024), repetitions=4,
                    k_values=(20, 100), seed=202675):
    """Return measured variation. Do NOT impose unvalidated thresholds."""
    gt = np.asarray(gt, dtype=float)
    pool = np.asarray(fresh_pool, dtype=float)
    sizes = tuple(int(n) for n in sample_sizes)
    if not sizes or min(sizes) < 2 or repetitions < 2:
        raise ValueError("Need sample sizes >=2 and at least 2 partitions")
    if len(pool) < max(sizes) * repetitions:
        raise ValueError("Insufficient independent pool for disjoint replicates")
    tree = cKDTree(gt)
    pool_distances = tree.query(pool, k=1)[0]
    if np.any(pool_distances <= 1e-12):
        raise ValueError("Candidate fresh sample overlaps finite GT")
    reference_gt_nn = float(np.median(tree.query(gt, k=2)[0][:, 1]))
    if reference_gt_nn <= 0:
        raise ValueError("Invalid archived nearest-neighbor GT spacing")
    raw_blur = np.asarray(raw_blur, dtype=float)
    if raw_blur.ndim != 1 or len(raw_blur) < 2 or np.any(~np.isfinite(raw_blur)):
        raise ValueError("Expected archived positive normalized blur ECDF")
    if np.any(raw_blur < 0):
        raise ValueError("Expected nonnegative uncorrected blur")
    # Observed distance distributions are fully frozen and independent of the
    # candidate epsilon. Both are generated *once* from the old frozen ECDF;
    # 0.35 and 0.65 avoid automatically making every Q=0 (unlike bad=blur).
    obs_raw = {str(f): f * raw_blur for f in (0.35, 0.65)}
    scale_by_k = {
        str(k): float(baselines.get_resolution_factor_k(gt, int(k), seed=0))
        for k in k_values
    }
    if np.any(np.asarray(list(scale_by_k.values())) <= 0):
        raise ValueError("Invalid positive FPS scale")

    record = {
        "method": "disjoint_subsets_from_one_deterministic_analytical_pool",
        "sampling_note": (
            "Disjoint within each N, but not independent analytical PF generator "
            "calls. This probes subsampling sensitivity and ordered-grid bias, "
            "not generalization to all analytical samplers/seeds."
        ),
        "reference_gt_nn_median": reference_gt_nn,
        "candidate_pool_count": len(pool),
        "candidate_pool_regions_f1": _region_hist(pool, gt),
        "baseline_first_1024": {},
        "sample_sizes": {},
        "warning_signals": [],
    }
    old_prefix = np.asarray(old_prefix, dtype=float)
    if len(old_prefix) != 1024:
        raise ValueError("Historical estimator requires 1024 fresh points")
    for k, scale in scale_by_k.items():
        e = estimate_ideal_residue(gt, old_prefix, scale)
        record["baseline_first_1024"][k] = {
            "epsilon": e,
            "q": {fraction: _fixed_q(raw, raw_blur, e, k=int(k))
                  for fraction, raw in obs_raw.items()},
        }
    record["baseline_first_1024"]["regions_f1"] = _region_hist(old_prefix, gt)

    rng = np.random.default_rng(seed)
    # Separate shuffle per sample size, disjoint groups within each size.
    # Note: groups across sizes reuse points, so inter-size measurements
    # are dependent; avoid treating them as separate independent experiments.
    for n in sizes:
        perm = rng.permutation(len(pool))
        repeats = []
        for j in range(repetitions):
            pts = pool[perm[j*n:(j+1)*n]]
            distances = tree.query(pts, k=1)[0]
            cover = cKDTree(pts).query(gt, k=1)[0]
            run = {
                "replicate": j,
                "n": int(n),
                "region_f1_counts": _region_hist(pts, gt),
                "local_GT_distances": {
                    "median": float(np.median(distances) / reference_gt_nn),
                    "p95": float(np.percentile(distances, 95) / reference_gt_nn),
                },
                "GT_to_sample_coverage": {
                    "median_over_GT_nn": float(np.median(cover) / reference_gt_nn),
                    "p95_over_GT_nn": float(np.percentile(cover, 95) / reference_gt_nn),
                },
                "k": {},
            }
            for k, scale in scale_by_k.items():
                epsilon = float(np.median(distances) / scale)
                run["k"][k] = {
                    "epsilon": epsilon,
                    "q": {fraction: _fixed_q(raw, raw_blur, epsilon, k=int(k))
                          for fraction, raw in obs_raw.items()},
                }
            repeats.append(run)
        summary = {}
        for k in scale_by_k:
            eps = [x["k"][k]["epsilon"] for x in repeats]
            summary[k] = {
                "epsilon": _stats(eps),
                "q": {fraction: _stats([
                    x["k"][k]["q"][fraction] for x in repeats
                ]) for fraction in obs_raw},
            }
            if n == 1024:
                earlier = record["baseline_first_1024"][k]["epsilon"]
                span = _stats(eps)
                summary[k]["legacy_prefix_over_subset_median"] = float(
                    earlier / max(span["median"], 1e-12)
                )
                summary[k]["legacy_prefix_outside_replicate_range"] = bool(
                    earlier < span["min"] or earlier > span["max"]
                )
                if summary[k]["legacy_prefix_outside_replicate_range"]:
                    record["warning_signals"].append(
                        f"N=1024 K={k}: legacy prefix epsilon outside "
                        "all disjoint-subset estimates"
                    )
        record["sample_sizes"][str(n)] = {
            "replicates": repeats,
            "summary": summary,
        }
    return record


def analyze_problem(name, source, *, sizes=(256, 512, 1024),
                    repetitions=4, pool_extra=8192, seed=202675):
    if name not in source["problems"]:
        raise ValueError(f"Missing frozen baseline for {name}")
    rec = source["problems"][name]
    path = ROOT / "moeabench/diagnostics/resources/references" / name / "calibration_package.npz"
    with np.load(path, allow_pickle=False) as package:
        gt = np.asarray(package["gt_norm"], dtype=float)
        if not np.array_equal(gt, np.asarray(rec["gt_reference"])):
            raise ValueError("Frozen GT does not match the baseline")
        ideal, nadir = np.asarray(package["ideal"]), np.asarray(package["nadir"])
    if np.any(nadir <= ideal):
        raise ValueError("Invalid calibration normalization")
    raw_blur = np.asarray(rec["20"]["closeness"]["rand_ecdf"], dtype=float)
    if not np.all(np.isfinite(raw_blur)):
        raise ValueError("Bad frozen reference ECDF")

    class NormalizedMop:
        def __init__(self, native):
            self.native = native
        def pf(self, n_points):
            return (np.asarray(self.native.pf(n_points=n_points)) - ideal) / (nadir-ideal)

    normal = NormalizedMop(getattr(mops, name)())
    # Crucial: retain ALL unique fresh candidates from the larger call.
    # The production helper intentionally truncates to its requested
    # n_fresh; doing that here would impose a *second* ordered-prefix bias
    # on a DPF5 grid before we attempt to measure the legacy bias.
    candidates = np.asarray(
        normal.pf(n_points=len(gt) + pool_extra), dtype=float
    )
    if candidates.ndim != 2 or candidates.shape[1] != gt.shape[1] or not np.all(np.isfinite(candidates)):
        raise ValueError(f"Invalid analytical candidate geometry for {name}")
    threshold = max(1e-12, 1e-10 * float(np.max(np.ptp(gt, axis=0))))
    d = cKDTree(gt).query(candidates, k=1)[0]
    pool = candidates[d > threshold]
    old = independent_analytical_reference(normal, gt, n_fresh=1024)
    if len(pool) < 2 or old is None:
        raise ValueError(f"Cannot construct independent analytical pool for {name}")
    if len(pool) < max(sizes)*repetitions:
        raise ValueError(f"{name}: only {len(pool)} pool points, need {max(sizes)*repetitions}")
    record = compare_subsets(
        gt, pool, old, raw_blur, sample_sizes=sizes,
        repetitions=repetitions, seed=seed,
    )
    record["problem"] = name
    record["archived_gt_shape"] = list(gt.shape)
    record["gt_sha256"] = hashlib.sha256(np.ascontiguousarray(gt).tobytes()).hexdigest()
    record["frozen_baseline_closeness_ecdf_k20_count"] = len(raw_blur)
    record["sample_sizes_n"] = list(sizes)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", action="append", dest="problems", required=True)
    parser.add_argument("--sample-size", action="append", type=int, dest="sizes")
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--pool-extra", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=202675)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error("Refusing to overwrite previous robustness experiment")
    if out.is_relative_to((ROOT / "calibration/reports").resolve()) or out.is_relative_to(
        (ROOT / "moeabench/diagnostics/resources").resolve()
    ):
        parser.error("Cannot overwrite certified calibration or reports")
    sizes = tuple(args.sizes or (256, 512, 1024))
    with (ROOT / "moeabench/diagnostics/resources/baselines_v0.14.0.json").open() as stream:
        source = json.load(stream)
    data = {"status": "NONCERTIFIED_SAMPLING_ROBUSTNESS_PILOT",
            "seed": args.seed, "repetitions": args.repetitions,
            "K": [20, 100], "results": {}, "failures": {}}
    for name in args.problems:
        try:
            result = analyze_problem(name, source, sizes=sizes,
                                     repetitions=args.repetitions,
                                     pool_extra=args.pool_extra, seed=args.seed)
            data["results"][name] = result
            print(json.dumps({
                "problem": name,
                "baseline_prefix_epsilon": result["baseline_first_1024"]["100"]["epsilon"],
                "replicate_1024_epsilon": result["sample_sizes"]["1024"]["summary"]["100"]["epsilon"],
                "q_sensitivity": result["sample_sizes"]["1024"]["summary"]["100"]["q"],
                "warning_signals": result["warning_signals"],
            }), flush=True)
        except (ValueError, TypeError, AttributeError, IndexError, OSError) as exc:
            data["failures"][name] = f"{type(exc).__name__}: {exc}"
            print(f"UNSUPPORTED {name}: {data['failures'][name]}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    if data["failures"]:
        raise SystemExit("Some robustness experiments failed; inspect artifact")


if __name__ == "__main__":
    main()
