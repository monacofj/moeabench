# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only DTLZ8 finite-GT policy decision evidence for M=2,3,5,10.

No candidate is auto-promoted by passing finite-probe diagnostics.
A dominance relation among sampling designs is evaluated separately
for each actual Pareto component, with three independent probe offsets.
"""
import argparse
import json
import numpy as np

from scripts.audit.gt_dtlz8_candidate_issue119 import candidate_objectives
from scripts.audit.gt_dtlz8_component_comparison_issue119 import compare
from scripts.audit.gt_dtlz8_membership_issue119 import (
    feasibility_residual, membership_residual,
)
from scripts.audit.gt_dtlz8_spacing_sensitivity_issue119 import component_spacing

METHODS = ("candidate", "lattice", "hybrid")


def summarize(m, budget, offsets=(11941, 11942, 11943), probes=512):
    if m not in (2, 3, 5, 10):
        raise ValueError("This policy comparison covers M=2,3,5,10.")
    if budget < m + 1:
        raise ValueError("Budget too small for all prescribed components.")
    if len(set(offsets)) != len(offsets) or len(offsets) < 3:
        raise ValueError("At least three distinct independent offsets required.")

    data = [compare(m, budget, probes_per_component=probes, offset=s)
            for s in offsets]
    checks = {}
    for name, strategy in (("candidate", "halton"), ("lattice", "lattice"),
                           ("hybrid", "hybrid")):
        front = candidate_objectives(m, budget, patch_strategy=strategy)
        spacing = component_spacing(front)
        checks[name] = {
            "count": len(front),
            "unique": len(np.unique(front, axis=0)),
            "max_pareto_residual": float(np.max(membership_residual(front))),
            "max_constraint_violation": float(np.max(feasibility_residual(front))),
            "component_spacing": spacing,
        }
        if checks[name]["unique"] != budget:
            raise AssertionError(f"{name}: cardinality failure")
        if checks[name]["max_pareto_residual"] > 1e-10:
            raise AssertionError(f"{name}: off-front sample")
        if checks[name]["max_constraint_violation"] > 1e-10:
            raise AssertionError(f"{name}: infeasible sample")

    comp_names = list(data[0]["components"])
    comparison = {}
    for component in comp_names:
        comparison[component] = {}
        for method in METHODS:
            maximum = [run["components"][component][method]["max_euclidean_gap"]
                       for run in data]
            mean = [run["components"][component][method]["mean_euclidean_gap"]
                    for run in data]
            comparison[component][method] = {
                "max_gap_by_offset": maximum,
                "mean_gap_by_offset": mean,
                "worst_max_gap": max(maximum),
                "worst_mean_gap": max(mean),
            }

    # A finite-probe non-regression policy requires no component
    # degradation in the maximum gap at any of the predeclared offsets.
    # This is a transparent comparison, not statistical certification.
    tolerance = 1e-12
    lattice_nonworse = all(
        comparison[c]["lattice"]["max_gap_by_offset"][i]
        <= comparison[c]["candidate"]["max_gap_by_offset"][i] + tolerance
        for c in comp_names for i in range(len(offsets))
    )
    halton_nonworse = all(
        comparison[c]["candidate"]["max_gap_by_offset"][i]
        <= comparison[c]["lattice"]["max_gap_by_offset"][i] + tolerance
        for c in comp_names for i in range(len(offsets))
    )
    hybrid_nonworse_halton = all(
        comparison[c]["hybrid"]["max_gap_by_offset"][i]
        <= comparison[c]["candidate"]["max_gap_by_offset"][i] + tolerance
        for c in comp_names for i in range(len(offsets))
    )
    hybrid_nonworse_lattice = all(
        comparison[c]["hybrid"]["max_gap_by_offset"][i]
        <= comparison[c]["lattice"]["max_gap_by_offset"][i] + tolerance
        for c in comp_names for i in range(len(offsets))
    )
    if lattice_nonworse and not halton_nonworse:
        probe_preference = "lattice"
    elif halton_nonworse and not lattice_nonworse:
        probe_preference = "halton"
    elif lattice_nonworse and halton_nonworse:
        probe_preference = "tie"
    else:
        probe_preference = "mixed_no_proven_winner"

    return {
        "M": m, "requested": budget, "offsets": list(offsets),
        "probe_points_per_component": probes,
        "incumbent_count": data[0]["incumbent_count"],
        "incumbent_unique": data[0]["incumbent_unique"],
        "admissibility": checks,
        "per_component": comparison,
        "finite_probe_max_gap_preference": probe_preference,
        "hybrid_no_worse_than_halton": hybrid_nonworse_halton,
        "hybrid_no_worse_than_lattice": hybrid_nonworse_lattice,
        "production_policy_approved": False,
        "canonical_mixed_dimensional_measure_defined": False,
        "continuous_coverage_certified": False,
        "incumbent_reused": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objectives", nargs="+", type=int, default=[2, 3, 5, 10])
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 512, 2000])
    parser.add_argument("--probes", type=int, default=512)
    parser.add_argument("--offsets", nargs="+", type=int,
                        default=[11941, 11942, 11943])
    args = parser.parse_args()
    results = [summarize(m, n, offsets=args.offsets, probes=args.probes)
               for m in args.objectives for n in args.budgets]
    print(json.dumps({
        "schema": "dtlz8_m2_m3_m5_m10_research_policy_v1",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
