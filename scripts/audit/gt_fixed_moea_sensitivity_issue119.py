# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only fixed-MOEA-front reference sensitivity, issue #119.

Each stochastic MOEA run happens exactly once. Its objective matrix is
then frozen in memory and measured against two independently generated
finite GT resolutions. This is NOT ADR 0056 performance certification;
it compares reference construction, not candidate software versions.
No calibration, cache, baseline or certificate is written.
"""
import argparse
import hashlib
import json
import os
import platform

import numpy as np
import moeabench as mb

from scripts.audit.gt_reference_sensitivity_issue119 import raw_igd_and_igd_plus


PROBLEMS = {
    "DTLZ2": lambda: mb.mops.DTLZ2(M=3),
    "DTLZ7": lambda: mb.mops.DTLZ7(M=3),
    "DTLZ8": lambda: mb.mops.DTLZ8(M=3),
    "DPF5": lambda: mb.mops.DPF5(M=3, D=2),
}
ALGORITHMS = {
    "NSGA2": mb.moeas.NSGA2,
    "SPEA2": mb.moeas.SPEA2,
}


def run_frozen_population(problem, algorithm, seed, population=32, generations=12):
    if problem not in PROBLEMS or algorithm not in ALGORITHMS:
        raise ValueError("Unknown benchmark or algorithm")
    mop = PROBLEMS[problem]()
    alg = ALGORITHMS[algorithm](
        population=population, generations=generations, seed=seed)
    exp = mb.experiment(mop=mop, moea=alg)
    exp.run(repeat=1, silent=True)
    front = np.asarray(exp.last_run.front(), dtype=np.float64)
    if front.ndim != 2 or front.shape[1] != 3 or not len(front):
        raise RuntimeError("Invalid final MOEA objective front")
    if not np.isfinite(front).all():
        raise RuntimeError("Nonfinite MOEA objectives")
    return front


def metric_snapshot(front, reference):
    """Evaluate the SAME front through the public metrics API.

    Public IGD and IGD+ are the production reference-normalized
    indicators. Raw values are retained only as separate diagnostics.
    Exact raw HV uses the finite GT as the fixed normalization context;
    it is not a calibrated absolute HV.
    """
    reference = np.asarray(reference, dtype=np.float64)
    igd = float(np.asarray(mb.metrics.igd(front, ref=reference).values)[-1, 0])
    plus = float(np.asarray(mb.metrics.igdplus(front, ref=reference).values)[-1, 0])
    hv = float(np.asarray(mb.metrics.hv(
        front, ref=reference, mode="exact", scale="raw",
        progress=False).values)[-1, 0])
    results = {
        "public_igd": igd, "public_igd_plus": plus,
        "public_hv_raw_reference_context": hv,
        **raw_igd_and_igd_plus(reference, front),
    }
    if not all(np.isfinite(v) for v in results.values()):
        raise RuntimeError("Nonfinite reference-dependent metric")
    return results


def evaluate_one(problem, algorithm, seed, population=32, generations=12,
                 reference_budgets=(128, 512)):
    front = run_frozen_population(
        problem, algorithm, seed, population, generations)
    mop = PROBLEMS[problem]()
    front_hash = hashlib.sha256(front.tobytes()).hexdigest()
    results = []
    for n in reference_budgets:
        reference = np.asarray(mop.pf(n_points=n), dtype=np.float64)
        scores = metric_snapshot(front, reference)
        results.append({
            "reference_budget": n,
            "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
            "scores": scores,
        })
    return {
        "problem": problem, "algorithm": algorithm, "seed": seed,
        "population": population, "generations": generations,
        "final_front_rows": len(front), "final_front_sha256": front_hash,
        "reference_evaluations": results,
    }


def analyze(records):
    """Count observed pairwise ordering changes, without a significance claim."""
    outcomes = []
    for problem in PROBLEMS:
        for seed in sorted({r["seed"] for r in records}):
            subset = {(r["algorithm"], r["seed"]): r for r in records
                      if r["problem"] == problem and r["seed"] == seed}
            if set(subset) != {(alg, seed) for alg in ALGORITHMS}:
                continue
            x, y = (subset[(name, seed)] for name in ALGORITHMS)
            budgets = [s["reference_budget"] for s in x["reference_evaluations"]]
            for metric in ("public_igd", "public_igd_plus",
                           "public_hv_raw_reference_context"):
                differences = [
                    a["scores"][metric]-b["scores"][metric]
                    for a, b in zip(x["reference_evaluations"],
                                    y["reference_evaluations"])
                ]
                # Exact equality is not interpreted as a direction.
                signs = [int(np.sign(z)) for z in differences]
                outcomes.append({
                    "problem": problem, "seed": seed, "metric": metric,
                    "budgets": budgets, "algorithm_difference": differences,
                    "signs": signs,
                    "ordering_reversed": signs[0]*signs[-1] < 0,
                })
    return outcomes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[11941, 11942])
    p.add_argument("--problems", nargs="+", choices=PROBLEMS,
                   default=list(PROBLEMS))
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--generations", type=int, default=12)
    args = p.parse_args()
    records = [
        evaluate_one(problem, algorithm, seed, args.population, args.generations)
        for problem in args.problems for algorithm in ALGORITHMS
        for seed in args.seeds
    ]
    print(json.dumps({
        "schema": "issue119_fixed_moea_reference_impact_v1",
        "source_sha": os.environ.get("GITHUB_SHA"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "records": records,
        "pairwise_orderings": analyze(records),
        "historical_reference_modified": False,
        "software_performance_certified": False,
        "absolute_hv_certified": False,
        "statistical_rank_change_certified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
