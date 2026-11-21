# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent finite-probe GT coverage for known analytical domains (#119).

Probes are generated from published decision-space conditions rather
than calling the production ps() sampler. Results are component-
specific for disconnected fronts. These are empirical nearest-
neighbor gaps, NOT certified continuous Hausdorff bounds.
"""
import argparse
import json
import numpy as np
from scipy.spatial import cKDTree

from moeabench.mops._dtlz7_pareto_regions import dtlz7_record_intervals
from scripts.audit.gt_postfix_quantitative_issue119 import make_problem


SUPPORTED = ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ7",
             "DTLZ9", "DPF1", "DPF2")


def independent_decision_probe(name, m, d, n, seed):
    rng = np.random.default_rng(seed)
    mop = make_problem(name, m=m, d=d)
    x = np.full((n, mop.N), 0.5)
    essential = d if name.startswith("DPF") else m
    if name in ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DPF1"):
        x[:, :essential-1] = rng.uniform(0, 1, (n, essential-1))
    elif name in ("DTLZ7", "DPF2"):
        (lo1, hi1), (lo2, hi2) = dtlz7_record_intervals()
        ids = np.arange(n) % (1 << (essential-1))
        rng.shuffle(ids)
        u = rng.uniform(0, 1, (n, essential-1))
        flags = (ids[:, None] >> np.arange(essential-1)) & 1
        x[:, :essential-1] = np.where(flags, lo2 + u*(hi2-lo2),
                                      lo1 + u*(hi1-lo1))
        x[:, essential-1:] = 0
    elif name == "DTLZ9":
        theta = rng.uniform(0, np.pi/2, n)
        f = np.column_stack((
            np.repeat(np.cos(theta)[:, None], m-1, axis=1),
            np.sin(theta)))
        x = np.zeros((n, mop.N))
        block = mop.N // m
        x[:, :block*m] = np.repeat(f**10, block, axis=1)
    else:
        raise ValueError(f"Unsupported independent probe: {name}")
    if name in ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DPF1"):
        x[:, essential-1:] = 0.5
    return x


def finite_probe_gap(name, budget, m=3, d=2, probes=512, seed=11941):
    if name not in SUPPORTED:
        raise ValueError(f"No independently validated probe for {name}")
    mop = make_problem(name, m=m, d=d)
    x = independent_decision_probe(name, m, d, probes, seed)
    reference = np.asarray(mop.evaluation(x)["F"])
    sample = np.asarray(mop.pf(n_points=budget))
    nearest = cKDTree(sample).query(reference, k=1, workers=1)[0]
    output = {"problem": name, "M": m, "D": d if name.startswith("DPF") else None,
              "budget": budget, "probe_count": probes, "seed": seed,
              "finite_max_gap": float(nearest.max()),
              "finite_mean_gap": float(nearest.mean()),
              "certified_continuous_bound": False}
    if name in ("DTLZ7", "DPF2"):
        essential = d if name == "DPF2" else m
        labels = (x[:, :essential-1] >= dtlz7_record_intervals()[1][0])
        ids = np.sum(labels.astype(np.int64) *
                     (1 << np.arange(essential-1)), axis=1)
        output["by_component"] = {
            str(int(k)): {"probes": int(np.count_nonzero(ids == k)),
                          "finite_max_gap": float(nearest[ids == k].max()),
                          "finite_mean_gap": float(nearest[ids == k].mean())}
            for k in np.unique(ids)}
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", nargs="+", type=int, default=[128, 512])
    p.add_argument("--objectives", nargs="+", type=int, default=[3, 5])
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[11941, 11942, 11943])
    p.add_argument("--probes", type=int, default=512)
    args = p.parse_args()
    results = []
    for name in SUPPORTED:
        for m in args.objectives:
            d = min(3, m-1) if name.startswith("DPF") else 2
            for budget in args.budgets:
                for seed in args.seeds:
                    results.append(finite_probe_gap(
                        name, budget, m=m, d=d,
                        probes=args.probes, seed=seed))
    print(json.dumps({
        "schema": "moeabench_issue119_independent_finite_probes_v1",
        "results": results,
        "excluded": ["DTLZ8", "DPF3", "DPF4", "DPF5"],
        "exclusion_reason": "Dedicated independent analytical oracles required",
        "certified_continuous_bound": False,
    }, indent=2))


if __name__ == "__main__":
    main()
