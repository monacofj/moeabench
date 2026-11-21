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


def independent_probes(m, n_per_component=256, offset=11941):
    """Generate line and patch interiors without invoking either sampler."""
    if m < 2 or n_per_component < 2:
        raise ValueError("Invalid objective count or probe resolution.")
    k = np.arange(1 + offset, 1 + offset + n_per_component)
    t = np.array([_radical_inverse(int(i), 5) for i in k])
    if m == 2:
        return {"line": np.column_stack(((1-t)/4, t))}
    line_t = 1/3 + t * (2/3)
    line = np.empty((n_per_component, m))
    line[:, :-1] = ((1-line_t)/4)[:, None]
    line[:, -1] = line_t
    out = {"line": line}
    u = np.array([_radical_inverse(int(i), 7) for i in k])
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


def compare(m, budget, probes_per_component=256, offset=11941):
    mop = DTLZ8(M=m)
    incumbent = np.asarray(mop.pf(n_points=budget), dtype=float)
    candidate = candidate_objectives(m, budget)
    probes = independent_probes(m, probes_per_component, offset)
    records = {}
    for name, reference in probes.items():
        records[name] = {
            "incumbent": _distance_statistics(reference, incumbent),
            "candidate": _distance_statistics(reference, candidate),
        }
    return {
        "M": m, "requested": budget,
        "incumbent_count": len(incumbent),
        "incumbent_unique": len(np.unique(incumbent, axis=0)),
        "candidate_count": len(candidate),
        "candidate_unique": len(np.unique(candidate, axis=0)),
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
