# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Research-only DTLZ8 spacing and IGD+ reference-sensitivity audit.

A finite Pareto reference is a measure on its *sampled points*, not a
canonical mixed-dimensional front measure. Accordingly this report
always labels which finite reference and independent population was
used; it never averages line and surface components without disclosure.
"""
import argparse
import json

import numpy as np
from scipy.spatial import cKDTree

from moeabench.mops.DTLZ8 import DTLZ8
from scripts.audit.gt_dtlz8_candidate_issue119 import candidate_objectives
from scripts.audit.gt_dtlz8_component_comparison_issue119 import (
    independent_probes, m4_preserving_completion,
)
from scripts.audit.gt_dtlz8_membership_issue119 import membership_residual


def _component_indices(f, tol=1e-10):
    f = np.asarray(f, dtype=float)
    m = f.shape[1]
    valid = membership_residual(f) <= tol
    labels = np.full(len(f), -1, dtype=int)  # -1 means off published PF
    if m == 2:
        labels[valid] = 0
    else:
        on_line = f[:, -1] >= (1.0 / 3.0 - tol)
        labels[valid & on_line] = 0
        on_patch = valid & ~on_line
        labels[on_patch] = 1 + np.argmin(f[on_patch, :-1], axis=1)
    return labels


def component_spacing(f):
    """Euclidean nearest-neighbor distances strictly within each component.

    The population counts and off-front exclusions remain explicit.
    Values from 1-D line and 2-D patches are NOT pooled into a single CV.
    """
    f = np.asarray(f, dtype=float)
    labels = _component_indices(f)
    records = {}
    for k in range(-1, f.shape[1] if f.shape[1] > 2 else 1):
        if k == -1:
            records["off_pareto"] = {"rows": int(np.sum(labels == -1))}
            continue
        subset = np.unique(f[labels == k], axis=0)
        name = "line" if k == 0 else f"patch_{k - 1}"
        if len(subset) < 2:
            records[name] = {"points": len(subset), "nn_cv": None,
                             "nn_mean": None, "nn_max": None}
            continue
        nearest = cKDTree(subset).query(subset, k=2, workers=1)[0][:, 1]
        mean = float(np.mean(nearest))
        records[name] = {
            "points": len(subset),
            "nn_mean": mean,
            "nn_cv": float(np.std(nearest) / mean) if mean > 0 else None,
            "nn_max": float(np.max(nearest)),
        }
    return records


def finite_igd_plus(reference, population):
    """Classical IGD+ of population against this finite reference (minimize)."""
    reference = np.asarray(reference, dtype=float)
    population = np.asarray(population, dtype=float)
    if not len(reference) or not len(population):
        raise ValueError("Reference and approximation must be nonempty.")
    result = []
    for block in np.array_split(reference, max(1, (len(reference) + 127)//128)):
        d = np.maximum(population[None, :, :] - block[:, None, :], 0.0)
        result.extend(np.min(np.linalg.norm(d, axis=2), axis=1).tolist())
    return float(np.mean(result))


def experiment(m, budget, seed=11941, population_per_region=32):
    if population_per_region < 2:
        raise ValueError("Population per region must be >=2.")
    original = np.asarray(DTLZ8(M=m).pf(n_points=budget))
    halton = candidate_objectives(m, budget)
    lattice = candidate_objectives(m, budget, patch_strategy="lattice")
    references = {"incumbent": original, "halton": halton, "lattice": lattice}
    if m == 4:
        references["preserving_completion"] = m4_preserving_completion(
            original, lattice, budget)

    # Independently generated Pareto-correct objective populations.
    probes = independent_probes(m, population_per_region, offset=seed)
    populations = {"line_only": probes["line"]}
    patch_names = [k for k in probes if k.startswith("patch_")]
    if patch_names:
        populations["one_patch_only"] = probes[patch_names[0]]
        # Explicit balanced-record sample, NOT an intrinsic measure.
        populations["equal_per_component"] = np.vstack(
            [probes[k] for k in probes])
    scores = {ref_name: {
        pop_name: finite_igd_plus(reference, population)
        for pop_name, population in populations.items()
    } for ref_name, reference in references.items()}
    return {
        "M": m, "budget": budget, "probe_seed": seed,
        "per_population_component_count": population_per_region,
        "reference_count": {k: len(v) for k, v in references.items()},
        "reference_spacing": {k: component_spacing(v) for k, v in references.items()},
        "igd_plus_by_finite_reference_and_population": scores,
        "rankings_are_reference_dependent": True,
        "canonical_mixed_dimensional_measure_defined": False,
        "certified_continuous_covering_bound": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objectives", nargs="+", type=int,
                        default=[2, 3, 4, 5, 10])
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11941, 11942, 11943])
    parser.add_argument("--population-per-region", type=int, default=32)
    args = parser.parse_args()
    records = [experiment(m, n, s, args.population_per_region)
               for m in args.objectives for n in args.budgets for s in args.seeds]
    print(json.dumps({"schema": "dtlz8_spacing_and_igd_reference_sensitivity_v1",
                      "records": records}, indent=2))


if __name__ == "__main__":
    main()
