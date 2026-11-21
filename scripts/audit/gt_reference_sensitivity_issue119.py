# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference-resolution sensitivity with fixed analytical probe populations.

This is raw Euclidean IGD / IGD+ for synthetic Pareto-correct
approximations, not the public reference-normalized indicator and not
a stochastic MOEA-performance comparison. No baseline is rewritten.
"""
import argparse
import json
import numpy as np
from scipy.spatial.distance import cdist

from scripts.audit.gt_independent_probes_issue119 import (
    SUPPORTED, independent_decision_probe,
)
from scripts.audit.gt_postfix_quantitative_issue119 import make_problem


def raw_igd_and_igd_plus(reference, population):
    reference = np.asarray(reference, dtype=float)
    population = np.asarray(population, dtype=float)
    if reference.ndim != 2 or population.ndim != 2:
        raise ValueError("Expected two-dimensional objective matrices.")
    if reference.shape[1] != population.shape[1] or not len(population):
        raise ValueError("Mismatched dimensions or empty approximation.")
    igd = float(cdist(reference, population).min(axis=1).mean())
    # Minimization: only approximation objectives worse than a reference
    # contribute to the IGD+ nearest dominated-octant distance.
    plus = []
    for block in np.array_split(reference, max(1, (len(reference)+127)//128)):
        diff = np.maximum(
            population[None, :, :] - block[:, None, :], 0)
        plus.extend(np.linalg.norm(diff, axis=2).min(axis=1))
    return {"raw_igd": igd, "raw_igd_plus": float(np.mean(plus))}


def examine(name, m=3, d=2, seed=11941, population_size=64,
            budgets=(128, 512, 2000)):
    if name not in SUPPORTED:
        raise ValueError("Missing an independent analytical approximation.")
    mop = make_problem(name, m=m, d=d)
    fixed_x = independent_decision_probe(
        name, m, d, population_size, seed)
    fixed_f = np.asarray(mop.evaluation(fixed_x)["F"])
    scores = []
    for budget in budgets:
        reference = np.asarray(mop.pf(n_points=budget))
        scores.append({"reference_budget": budget,
                       **raw_igd_and_igd_plus(reference, fixed_f)})
    return {
        "problem": name, "M": m, "D": d if name.startswith("DPF") else None,
        "population_seed": seed, "fixed_population_size": population_size,
        "reference_scores": scores,
        "population_type": "independent analytical synthetic probes",
        "production_normalization": False,
        "actual_moea_performance_test": False,
        "hv_test": False,
        "historical_baseline_unchanged": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objectives", nargs="+", type=int, default=[3, 5])
    p.add_argument("--budgets", nargs="+", type=int,
                   default=[128, 512, 2000])
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[11941, 11942, 11943])
    p.add_argument("--population", type=int, default=64)
    args = p.parse_args()
    records = []
    for name in SUPPORTED:
        for m in args.objectives:
            d = min(3, m-1) if name.startswith("DPF") else 2
            for seed in args.seeds:
                budgets = [
                    (10000 if name.startswith("DPF") else b)
                    if b == 2000 else b for b in args.budgets
                ]
                records.append(examine(name, m, d, seed,
                                       args.population, budgets))
    print(json.dumps({
        "schema": "moeabench_issue119_reference_sensitivity_v1",
        "records": records,
        "production_indicator_certification": False,
    }, indent=2))


if __name__ == "__main__":
    main()
