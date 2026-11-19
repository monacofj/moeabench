# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Research-only complete-cost / reference-size sensitivity study for #119.

Same generated GT arrays evaluated against independently drawn references
of multiple sizes. The instrumented wall-clock intervals include candidate
generation, inverse PS reconstruction, and forward MOP evaluation; reference
generation / diagnostic distances are excluded from generation timings.
No production MOP or baseline is modified.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from scripts.audit.gt_geometry_pilot import _MOPS, independent_reference, geometry_stats
from scripts.audit.gt_fps_preimage_study import analytical_preimages
from scripts.audit.gt_fps_comparison import farthest_candidates
from scripts.audit.gt_voronoi_controls_issue119 import (
    with_analytical_extrema, projected_mc_cvt,
)


def measure(name: str, m: int, n: int, *, pool_size: int,
            reference_sizes: tuple[int, ...], seed: int,
            iterations: int = 12) -> dict:
    if name not in ("DTLZ1", "DTLZ2") or m < 2 or n < m or pool_size < n:
        raise ValueError("Require supported geometry and pool>=n>=M>=2")
    if not reference_sizes or any(v < 2 for v in reference_sizes):
        raise ValueError("Reference sizes must all be >=2")
    mop = _MOPS[name](M=m)
    arrays = {}
    times = {}
    errors = {}

    def check(candidate: np.ndarray, variant: str) -> np.ndarray:
        x = analytical_preimages(mop, candidate)
        f = np.asarray(mop.evaluation(x)["F"], dtype=np.float64)
        errors[variant] = float(np.max(np.abs(f - candidate)))
        return f

    begin = time.perf_counter()
    original = np.asarray(mop.pf(n_points=n), dtype=np.float64)
    times["incumbent"] = time.perf_counter() - begin
    arrays["incumbent"] = original

    begin = time.perf_counter()
    original_for_axes = np.asarray(mop.pf(n_points=n), dtype=np.float64)
    arrays["anchored"] = check(with_analytical_extrema(original_for_axes, name),
                              "anchored")
    times["anchored"] = time.perf_counter() - begin

    begin = time.perf_counter()
    fps = farthest_candidates(name, m, n, candidates=pool_size, seed=seed)
    arrays["fps"] = check(fps, "fps")
    times["fps"] = time.perf_counter() - begin

    begin = time.perf_counter()
    base_for_cvt = np.asarray(mop.pf(n_points=n), dtype=np.float64)
    pool = independent_reference(name, pool_size, m, seed)
    init = with_analytical_extrema(base_for_cvt, name)
    cvt = projected_mc_cvt(name, init, pool, iterations=iterations)
    arrays["anchored_mc_cvt"] = check(cvt, "anchored_mc_cvt")
    times["anchored_mc_cvt"] = time.perf_counter() - begin

    begin = time.perf_counter()
    fps2 = farthest_candidates(name, m, n, candidates=pool_size, seed=seed)
    pool2 = independent_reference(name, pool_size, m, seed)
    hybrid = projected_mc_cvt(name, fps2, pool2, iterations=iterations)
    arrays["fps_mc_cvt"] = check(hybrid, "fps_mc_cvt")
    times["fps_mc_cvt"] = time.perf_counter() - begin

    # The generated GT arrays are held fixed across *all* reference sizes.
    observations = {}
    for ref_size in reference_sizes:
        reference = independent_reference(name, ref_size, m,
                                          seed + 1_000_003)
        observations[str(ref_size)] = {
            method: geometry_stats(points, reference, name)
            for method, points in arrays.items()
        }

    return {
        "name": name, "M": m, "budget": n, "candidate_pool": pool_size,
        "seed": seed, "iterations": iterations,
        "reference_sizes": list(reference_sizes),
        "generated_gt_is_fixed_across_reference_sizes": True,
        "full_generation_seconds": times,
        "inverse_error": errors,
        "reference_diagnostics": observations,
    }


def summarize(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        key = (record["name"], record["M"], record["budget"],
               record["candidate_pool"])
        groups.setdefault(key, []).append(record)
    result = []
    for (name, m, n, pool_size), rows in sorted(groups.items()):
        ref_sizes = rows[0]["reference_sizes"]
        metrics = {}
        methods = rows[0]["full_generation_seconds"]
        for method in methods:
            costs = [row["full_generation_seconds"][method] for row in rows]
            measurements = {}
            for sz in ref_sizes:
                v = [row["reference_diagnostics"][str(sz)][method] for row in rows]
                measurements[str(sz)] = {
                    k: float(np.median([d[k] for d in v]))
                    for k in ("reference_to_front_mean", "reference_to_front_max",
                              "reference_to_front_p95")
                }
            metrics[method] = {
                "complete_generation_median_seconds": float(np.median(costs)),
                "reference_sizes": measurements,
            }
        result.append({
            "name": name, "M": m, "budget": n, "candidate_pool": pool_size,
            "paired_seeds": len(rows), "methods": metrics,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--pool", type=int, default=4096)
    parser.add_argument("--reference-sizes", nargs="+", type=int,
                        default=[2048, 8192, 32768])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--iterations", type=int, default=12)
    a = parser.parse_args()
    records = [
        measure(name, m, n, pool_size=a.pool,
                reference_sizes=tuple(a.reference_sizes),
                seed=seed, iterations=a.iterations)
        for name in ("DTLZ1", "DTLZ2") for m in a.objectives
        for n in a.budgets for seed in a.seeds
    ]
    print(json.dumps({
        "schema": "gt119_complete_cost_ref_sensitivity_v1",
        "status": "research only, no continuous-covering certificate",
        "records": records, "summary": summarize(records),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
