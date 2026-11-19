# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent fixed-budget DTLZ1/2 comparison with anchored controls and Voronoi.

Research-only: no production PS/PF, calibrated GTs or baselines are modified.
Monte Carlo centroidal Voronoi (MC-CVT) uses independent uniform surface
candidates; projected cell means are an *approximate* Lloyd iteration, not
a proved continuous manifold CVT or minimum covering-radius solution.

Run from repository root:
  python -m scripts.audit.gt_voronoi_controls_issue119 --objectives 3 5 10
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from scripts.audit.gt_fps_preimage_study import analytical_preimages
from scripts.audit.gt_fps_comparison import farthest_candidates
from scripts.audit.gt_geometry_pilot import _MOPS, geometry_stats, independent_reference


def with_analytical_extrema(points: np.ndarray, name: str) -> np.ndarray:
    """Replace at most M rows with exact front axes; preserve point budget."""
    x = np.array(points, dtype=float, copy=True)
    if x.ndim != 2 or x.shape[0] < x.shape[1]:
        raise ValueError("Require n>=M and a 2D point matrix")
    x[:x.shape[1]] = (0.5 if name == "DTLZ1" else 1.0) * np.eye(x.shape[1])
    return x


def projected_mc_cvt(name: str, initial: np.ndarray, candidates: np.ndarray,
                     *, iterations: int = 12) -> np.ndarray:
    """Approximate centroidal Voronoi relaxation, keeping the M axes fixed.

    Assignment uses Euclidean nearest-center distances. The means of the
    assigned independent objective-space candidate points are projected
    to the simplex/hypersphere. An empty cell retains its current center.
    Extremes are anchored by construction, never reselected from candidates.
    """
    p = np.asarray(initial, dtype=float).copy()
    pool = np.asarray(candidates, dtype=float)
    if p.ndim != 2 or pool.ndim != 2 or p.shape[1] != pool.shape[1]:
        raise ValueError("Expected (n,M) centers and (C,M) candidates")
    n, m = p.shape
    if n < m or len(pool) < n or iterations < 0:
        raise ValueError("Require n>=M, candidate pool>=n, iterations>=0")
    target = 0.5 if name == "DTLZ1" else 1.0
    p[:m] = target * np.eye(m)
    for _ in range(iterations):
        labels = cKDTree(p).query(pool, k=1, workers=1)[1]
        sums = np.zeros_like(p)
        counts = np.bincount(labels, minlength=n)
        np.add.at(sums, labels, pool)
        active = counts[m:] > 0
        if np.any(active):
            cellmeans = sums[m:][active] / counts[m:][active, None]
            if name == "DTLZ1":
                cellmeans = target * cellmeans / cellmeans.sum(axis=1, keepdims=True)
            else:
                cellmeans = cellmeans / np.linalg.norm(cellmeans, axis=1, keepdims=True)
            p[m:][active] = cellmeans
    return p


def study(name: str, m: int, budget: int, *, pool_size: int,
          reference_size: int, seed: int, iterations: int) -> dict:
    if name not in ("DTLZ1", "DTLZ2"):
        raise ValueError("Compare distinct simplex/sphere geometries only")
    if m < 2 or budget < m or pool_size < budget or reference_size < 2:
        raise ValueError("Invalid dimension or sample sizes")
    mop = _MOPS[name](M=m)
    reference = independent_reference(name, reference_size, m, seed + 1_000_003)
    candidates = independent_reference(name, pool_size, m, seed)
    outputs = {}
    times = {}
    reprojection_error = {}
    variants = ["incumbent", "anchored", "fps", "anchored_mc_cvt", "fps_mc_cvt"]
    start = time.perf_counter()
    incumbent = np.asarray(mop.pf(n_points=budget), dtype=float)
    times["incumbent"] = time.perf_counter() - start
    anchored = with_analytical_extrema(incumbent, name)
    start = time.perf_counter()
    fps = farthest_candidates(name, m, budget, candidates=pool_size, seed=seed)
    times["fps"] = time.perf_counter() - start
    start = time.perf_counter()
    cvt = projected_mc_cvt(name, anchored, candidates, iterations=iterations)
    times["anchored_mc_cvt"] = time.perf_counter() - start
    start = time.perf_counter()
    hybrid = projected_mc_cvt(name, fps, candidates, iterations=iterations)
    times["fps_mc_cvt"] = time.perf_counter() - start
    times["anchored"] = 0.0  # only replacement, not the incumbent generation
    for variant, x in zip(variants, [incumbent, anchored, fps, cvt, hybrid]):
        # All experimental samples must admit a valid PS inverse.
        if variant != "incumbent":
            dec = analytical_preimages(mop, x)
            f = np.asarray(mop.evaluation(dec)["F"], dtype=float)
            reprojection_error[variant] = float(np.max(np.abs(f - x)))
            x = f
        outputs[variant] = geometry_stats(x, reference, name)
    return {"name": name, "M": m, "budget": budget, "pool": pool_size,
            "reference_size": reference_size, "seed": seed, "iterations": iterations,
            "stats": outputs, "seconds": times, "inverse_errors": reprojection_error}


def summarize(records: list[dict]) -> list[dict]:
    groups = {}
    for r in records:
        groups.setdefault((r["name"], r["M"], r["budget"], r["pool"]), []).append(r)
    summary = []
    for key, group in sorted(groups.items()):
        metrics = {}
        for variant in group[0]["stats"]:
            metrics[variant] = {
                field: float(np.median([r["stats"][variant][field] for r in group]))
                for field in ("reference_to_front_mean", "reference_to_front_max",
                              "nearest_neighbor_cv", "extrema_max_distance")
            }
            metrics[variant]["seconds_median"] = float(np.median([
                r["seconds"][variant] for r in group]))
        summary.append({"name": key[0], "M": key[1], "budget": key[2],
                        "pool": key[3], "seeds": len(group), "methods": metrics})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--pool", type=int, default=4096)
    parser.add_argument("--reference-size", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    a = parser.parse_args()
    records = [study(name, m, n, pool_size=a.pool, reference_size=a.reference_size,
                     seed=seed, iterations=a.iterations)
               for name in ("DTLZ1", "DTLZ2") for m in a.objectives
               for n in a.budgets for seed in a.seeds]
    print(json.dumps({"schema": "gt119_mc_cvt_controls_v1",
                      "note": "exploratory, no continuous covering guarantee",
                      "records": records, "summary": summarize(records)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
