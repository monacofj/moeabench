# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent finite-probe DTLZ8 component coverage / IGD+ diagnostics.

The probes have explicit *per-component* coordinates, deliberately
different from the candidate radical-inverse bases. No mixed-dimensional
component weights or global reference probability measure are implied.
"""
import argparse
import json

import numpy as np
from scipy.spatial import cKDTree

from moeabench.mops.DTLZ8 import DTLZ8
from scripts.audit.gt_dtlz8_candidate_issue119 import (
    _radical_inverse, candidate_objectives,
)
from scripts.audit.gt_dtlz8_membership_issue119 import membership_residual


def independent_probes(m, n_per_component=256, offset=11941):
    """Generate line and patch interiors without invoking either sampler."""
    if m < 2 or n_per_component < 2:
        raise ValueError("Invalid objective count or probe resolution.")
    k = np.arange(1 + offset, 1 + offset + n_per_component)
    # A seed-specific irrational rotation avoids almost-identical probe
    # sets from adjacent integer starting indices.
    t_shift = (offset * ((np.sqrt(5.0) - 1) / 2)) % 1
    u_shift = (offset * ((np.sqrt(3.0) - 1) / 2)) % 1
    t = (np.array([_radical_inverse(int(i), 5) for i in k]) + t_shift) % 1
    if m == 2:
        return {"line": np.column_stack(((1-t)/4, t))}
    line_t = 1/3 + t * (2/3)
    line = np.empty((n_per_component, m))
    line[:, :-1] = ((1-line_t)/4)[:, None]
    line[:, -1] = line_t
    out = {"line": line}
    u = (np.array([_radical_inverse(int(i), 7) for i in k]) + u_shift) % 1
    patch_t = t/3
    a = (1-patch_t)/4
    b = 1-2*patch_t
    low = a + u*(b/2-a)
    high = b-low
    for j in range(m-1):
        patch = np.repeat(high[:, None], m, axis=1)
        patch[:, j] = low
        patch[:, -1] = patch_t
        out[f"patch_{j}"] = patch
    return out


def _distance_statistics(reference, approximation):
    """Nearest point in approximation for each independent reference point."""
    approximation = np.asarray(approximation, dtype=float)
    distances = cKDTree(approximation).query(reference, workers=1)[0]
    # Positive-part Euclidean metric: IGD+ with fixed independent probes.
    best = np.full(len(reference), np.inf)
    for start in range(0, len(reference), 128):
        rows = reference[start:start+128]
        positive = np.maximum(
            approximation[None, :, :] - rows[:, None, :], 0.0)
        best[start:start+len(rows)] = np.linalg.norm(
            positive, axis=2).min(axis=1)
    return {
        "max_euclidean_gap": float(np.max(distances)),
        "mean_euclidean_gap": float(np.mean(distances)),
        "mean_igd_plus_positive_part": float(np.mean(best)),
    }



def m4_preserving_completion(incumbent, lattice, budget, *, tolerance=1e-10):
    """Research control: retain every distinct valid incumbent point for M=4.

    A fixed-budget superset of all incumbent unique objectives cannot
    worsen the nearest-approximation distance for any reference point.
    This is a covering guarantee, not a regularity/measure guarantee.
    """
    incumbent = np.asarray(incumbent, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    if incumbent.ndim != 2 or incumbent.shape[1] != 4:
        raise ValueError("Completion is currently validated only for M=4.")
    if lattice.shape != (budget, 4):
        raise ValueError("Lattice must contain exactly the requested budget.")
    if not np.isfinite(incumbent).all():
        raise ValueError("Incumbent contains nonfinite values.")
    if np.max(membership_residual(incumbent)) > tolerance:
        raise ValueError("Incumbent contains off-Pareto objective rows.")
    # Retain the original ordering for the first occurrence of each row.
    _, indices = np.unique(incumbent, axis=0, return_index=True)
    unique_incumbent = incumbent[np.sort(indices)]
    if len(unique_incumbent) > budget:
        raise ValueError("Incumbent already exceeds the requested budget.")
    if len(unique_incumbent) == budget:
        return unique_incumbent.copy()
    # Select additional points farthest from the preserved reference,
    # rather than filling in lattice row order (which overfills the line).
    # Greedy distances are measured in the native objective space.
    candidates = np.unique(lattice, axis=0)
    if len(candidates) + len(unique_incumbent) < budget:
        raise ValueError("Insufficient distinct candidates to fill budget.")
    minimum = cKDTree(unique_incumbent).query(candidates, k=1)[0] ** 2
    selected = []
    for _ in range(budget - len(unique_incumbent)):
        chosen = int(np.argmax(minimum))
        if minimum[chosen] <= 0:
            raise ValueError("Insufficient distinct candidates to fill budget.")
        point = candidates[chosen]
        selected.append(point)
        new_distance = np.sum((candidates - point) ** 2, axis=1)
        minimum = np.minimum(minimum, new_distance)
        minimum[chosen] = -1.0  # disallow selecting this point again
    completed = np.vstack((unique_incumbent, np.asarray(selected)))
    if len(completed) != budget or len(np.unique(completed, axis=0)) != budget:
        raise ValueError("Completion failed unique-budget contract.")
    return completed


def compare(m, budget, probes_per_component=256, offset=11941):
    mop = DTLZ8(M=m)
    incumbent = np.asarray(mop.pf(n_points=budget), dtype=float)
    candidate = candidate_objectives(m, budget)
    lattice = candidate_objectives(m, budget, patch_strategy="lattice")
    hybrid = candidate_objectives(m, budget, patch_strategy="hybrid")
    completed_m4 = (m4_preserving_completion(incumbent, lattice, budget)
                    if m == 4 else None)
    probes = independent_probes(m, probes_per_component, offset)
    records = {}
    for name, reference in probes.items():
        records[name] = {
            "incumbent": _distance_statistics(reference, incumbent),
            "candidate": _distance_statistics(reference, candidate),
            "lattice": _distance_statistics(reference, lattice),
            "hybrid": _distance_statistics(reference, hybrid),
        }
        if completed_m4 is not None:
            records[name]["m4_completion"] = _distance_statistics(
                reference, completed_m4)
    return {
        "M": m, "requested": budget,
        "incumbent_count": len(incumbent),
        "incumbent_unique": len(np.unique(incumbent, axis=0)),
        "candidate_count": len(candidate),
        "candidate_unique": len(np.unique(candidate, axis=0)),
        "lattice_count": len(lattice),
        "lattice_unique": len(np.unique(lattice, axis=0)),
        "hybrid_count": len(hybrid),
        "hybrid_unique": len(np.unique(hybrid, axis=0)),
        "m4_completion_count": len(completed_m4) if completed_m4 is not None else None,
        "m4_completion_unique": (len(np.unique(completed_m4, axis=0))
                                 if completed_m4 is not None else None),
        "probe_offset": offset, "probe_count_per_component": probes_per_component,
        "components": records,
        "continuous_coverage_certified": False,
        "canonical_mixed_dimensional_measure_chosen": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objectives", type=int, nargs="+", default=[2, 3, 4, 5, 10])
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11941, 11942, 11943])
    parser.add_argument("--probes", type=int, default=256)
    args = parser.parse_args()
    results = [compare(m, n, args.probes, seed)
               for m in args.objectives
               for n in args.budgets
               for seed in args.seeds]
    print(json.dumps({"schema": "dtlz8_issue119_component_probe_v1",
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
