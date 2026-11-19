# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #119: fixed-real-MOEA metric sensitivity to analytical GT discretization.

Research only. Every NSGA-II run is executed ONCE for each MOP/seed; its
objectives are held bitwise fixed while reference front methods/budgets vary.
IGD/IGD+ use MoeaBench's actual metric classes; HV uses the actual
GEN_hypervolume class with fixed analytical ideal/nadir, and the absolute
HV ratio uses the same H_sol/H_GT rule as evaluator.hypervolume(scale='abs').
This DOES NOT call calibrate(), update sidecars or change release baselines.

Invoked from repository root with python -m scripts.audit.gt_metric_sensitivity_issue119.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from moeabench.metrics.GEN_igd import GEN_igd
from moeabench.metrics.GEN_igdplus import GEN_igdplus
from moeabench.metrics.GEN_hypervolume import GEN_hypervolume
from scripts.audit.gt_fps_comparison import farthest_candidates
from scripts.audit.gt_fps_preimage_study import analytical_preimages
from scripts.audit.gt_geometry_pilot import _MOPS, independent_reference
from scripts.audit.gt_voronoi_controls_issue119 import projected_mc_cvt


class _MopProblem(Problem):
    def __init__(self, mop):
        self.mop = mop
        super().__init__(n_var=mop.N, n_obj=mop.M, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.mop.evaluation(X)["F"]


def frozen_moea_output(mop, seed: int, *, pop_size: int, generations: int):
    """Execute a real NSGA-II once, return a read-only objective array."""
    if pop_size < 4 or generations < 2:
        raise ValueError("Need population>=4 and generations>=2")
    result = minimize(
        _MopProblem(mop), NSGA2(pop_size=pop_size),
        termination=("n_gen", generations),
        seed=seed, verbose=False, save_history=False,
    )
    front = np.asarray(result.pop.get("F"), dtype=np.float64).copy()
    if front.ndim != 2 or front.shape != (pop_size, mop.M):
        raise ValueError("MOEA returned an unexpected final population")
    if not np.all(np.isfinite(front)):
        raise ValueError("MOEA output has invalid objective values")
    front.setflags(write=False)
    digest = hashlib.sha256(front.tobytes()).hexdigest()
    return front, digest


def gt_variants(mop, n: int, *, seed: int, pool_size: int,
                iterations: int) -> dict:
    """Generate independent analytical GT candidates; never call calibrate."""
    if n < mop.M or pool_size < n:
        raise ValueError("Require pool>=n>=M")
    name = mop.name
    current = np.asarray(mop.pf(n_points=n), dtype=float)
    fps = farthest_candidates(name, mop.M, n, candidates=pool_size, seed=seed)
    pool = independent_reference(name, pool_size, mop.M, seed)
    hybrid = projected_mc_cvt(name, fps, pool, iterations=iterations)
    choices = {"incumbent": current, "fps": fps, "fps_mc_cvt": hybrid}
    verified = {}
    for key, points in choices.items():
        if key == "incumbent":
            verified[key] = points
        else:
            decision = analytical_preimages(mop, points)
            mapped = np.asarray(mop.evaluation(decision)["F"], dtype=float)
            if not np.allclose(mapped, points, atol=1e-10, rtol=0):
                raise ValueError("GT candidate inverse mapping failed")
            verified[key] = mapped
    return verified


def compute_metrics(front: np.ndarray, gt: np.ndarray, *,
                    target: float) -> dict:
    """Use original metric implementations with common fixed HV bounds."""
    m = front.shape[1]
    bounds_min = np.zeros(m)
    bounds_max = np.full(m, target)
    igd = float(GEN_igd([front], gt).evaluate()[0])
    igdplus = float(GEN_igdplus([front], gt).evaluate()[0])
    hv_sol = float(GEN_hypervolume(
        [front], m, bounds_min, bounds_max).evaluate()[0])
    if hv_sol <= 0 or not np.isfinite(hv_sol):
        raise ValueError("MOEA has zero HV under fixed analytical bounds; increase MOEA generations or reconsider study conditions")
    hv_gt = float(GEN_hypervolume(
        [gt], m, bounds_min, bounds_max).evaluate()[0])
    if hv_gt <= 0 or not np.isfinite(hv_gt):
        raise ValueError("Invalid GT denominator in absolute HV analysis")
    return {
        "igd": igd, "igdplus": igdplus,
        "hv_solution_fixed_bounds": hv_sol,
        "hv_gt_fixed_bounds": hv_gt,
        "hv_absolute_ratio_fixed_bounds": hv_sol / hv_gt,
    }


def run_case(name: str, m: int, *, moea_seed: int, budgets: tuple[int, ...],
             pool_size: int, iterations: int, pop_size: int,
             generations: int) -> dict:
    if name not in ("DTLZ1", "DTLZ2") or m < 2:
        raise ValueError("Only the independently validated simplex/sphere MOPs")
    if not budgets or min(budgets) < m or max(budgets) > pool_size:
        raise ValueError("Bad GT budgets")
    mop = _MOPS[name](M=m)
    front, digest = frozen_moea_output(
        mop, moea_seed, pop_size=pop_size, generations=generations)
    objective_snapshot = front.copy()
    reference_seed = moea_seed + 1000
    target = 0.5 if name == "DTLZ1" else 1.0
    measurements = {}
    for n in budgets:
        candidate_sets = gt_variants(
            mop, n, seed=reference_seed, pool_size=pool_size,
            iterations=iterations)
        measurements[str(n)] = {
            label: compute_metrics(front, gt, target=target)
            for label, gt in candidate_sets.items()
        }
    np.testing.assert_array_equal(front, objective_snapshot)
    assert hashlib.sha256(front.tobytes()).hexdigest() == digest
    return {
        "problem": name, "M": m, "moea": "pymoo.NSGA2",
        "moea_seed": moea_seed,
        "moea_population": pop_size,
        "moea_generations": generations,
        "final_population_sha256": digest,
        "reference_seed": reference_seed, "gt_budgets": list(budgets),
        "pool": pool_size, "metrics": measurements,
    }


def summarize(records: list[dict], *, largest_budget: int):
    """Measured changes between GT resolutions for identical MOEA data."""
    groups = {}
    for r in records:
        groups.setdefault((r["problem"], r["M"]), []).append(r)
    rows = []
    for (name, m), batch in sorted(groups.items()):
        values = {}
        for method in batch[0]["metrics"][str(largest_budget)]:
            reference = [r["metrics"][str(largest_budget)][method]
                         for r in batch]
            values[method] = {}
            for n in batch[0]["gt_budgets"]:
                rels = {}
                for metric in ("igd", "igdplus", "hv_gt_fixed_bounds",
                               "hv_absolute_ratio_fixed_bounds"):
                    current = [r["metrics"][str(n)][method][metric]
                               for r in batch]
                    base = [x[metric] for x in reference]
                    abs_d = [abs(a-b) for a,b in zip(current,base)]
                    rels[metric] = {
                        "median_abs_change_vs_largest": float(np.median(abs_d)),
                        "max_abs_change_vs_largest": float(max(abs_d)),
                    }
                values[method][str(n)] = rels
        rows.append({"problem": name, "M": m, "moea_seeds": len(batch),
                     "reference_budget": largest_budget, "methods": values})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objectives", nargs="+", type=int, default=[3])
    p.add_argument("--budgets", nargs="+", type=int, default=[128, 512, 1024])
    p.add_argument("--seeds", nargs="+", type=int, default=[101, 202])
    p.add_argument("--pool", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--pop-size", type=int, default=64)
    p.add_argument("--generations", type=int, default=35)
    a = p.parse_args()
    budgets = tuple(sorted(set(a.budgets)))
    records = [run_case(name, m, moea_seed=seed, budgets=budgets,
                        pool_size=a.pool, iterations=a.iterations,
                        pop_size=a.pop_size, generations=a.generations)
               for name in ("DTLZ1", "DTLZ2") for m in a.objectives
               for seed in a.seeds]
    print(json.dumps({
        "schema": "gt119_fixed_real_moea_metric_sensitivity_v1",
        "status": "exploratory; not an accuracy certification",
        "records": records,
        "summary": summarize(records, largest_budget=max(budgets)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
