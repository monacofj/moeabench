# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonmutating DPF3/DPF5 finite-sample diagnostics for issue #119.

Checks how the *actual* PS points evaluate and how the returned PF
differs from those points. Within-sample nondominance is not a global
Pareto-optimality certificate. Branch membership is in decision space.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial import cKDTree
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from moeabench.mops.DPF3 import DPF3
from moeabench.mops.DPF5 import DPF5


def _nd_count(values: np.ndarray) -> int:
    return len(NonDominatedSorting().do(values, only_non_dominated_front=True))


def inspect(name: str, m: int, budget: int, d: int = 2) -> dict:
    if name not in ("DPF3", "DPF5") or m <= d or d < 2 or budget < 2:
        raise ValueError("Expected DPF3/DPF5, M>D>=2, budget>=2")
    cls = DPF3 if name == "DPF3" else DPF5
    mop = cls(M=m, D=d)
    ps = np.asarray(mop.ps(n_points=budget), dtype=float)
    evaluated = np.asarray(mop.evaluation(ps)["F"], dtype=float)
    pf = np.asarray(mop.pf(n_points=budget), dtype=float)
    if (ps.ndim != 2 or evaluated.shape != (len(ps), m)
            or pf.ndim != 2 or pf.shape[1] != m
            or not np.isfinite(evaluated).all()
            or not np.isfinite(pf).all()):
        raise ValueError("Unexpected or nonfinite MOP output")
    dist, _ = cKDTree(evaluated).query(pf, k=1)
    result = {
        "name": name, "M": m, "D": d,
        "requested": budget, "ps_count": len(ps), "pf_count": len(pf),
        "pf_unique_count": len(np.unique(pf, axis=0)),
        "raw_evaluated_unique_count": len(np.unique(evaluated, axis=0)),
        "raw_evaluated_internally_dominated": len(evaluated) - _nd_count(evaluated),
        "returned_pf_internally_dominated": len(pf) - _nd_count(pf),
        "returned_pf_nearest_raw_max": float(dist.max()),
        "globally_pareto_optimal_certified": False,
        "globally_complete_certified": False,
    }
    if name == "DPF3":
        # The override filters objective vectors. Matching is by
        # distance, never by row index, since order/count may change.
        result["pf_subset_of_evaluated_within_1e_10"] = bool(np.all(dist <= 1e-10))
        result["raw_internally_nondominated_count"] = _nd_count(evaluated)
        result["pf_is_a_filtered_sample"] = len(pf) < len(evaluated)
    else:
        # The original branch condition is strict at x_1 = 1/3.
        branch_d = ps[:, 0] > (1.0 / 3.0)
        parts = {}
        for key, mask in (("P", ~branch_d), ("D", branch_d)):
            f = evaluated[mask]
            parts[key] = {
                "decision_count": int(mask.sum()),
                "unique_objective_count": int(len(np.unique(f, axis=0))),
                "internally_dominated_within_branch": int(len(f) - _nd_count(f)) if len(f) else 0,
                "objective_duplicates_within_branch": int(len(f) - len(np.unique(f, axis=0))),
            }
        result["branches"] = parts
        result["grid_side_if_M3"] = int(np.sqrt(budget)) if m == 3 else None
        result["raw_vs_returned_count_equal"] = len(evaluated) == len(pf)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 512])
    args = parser.parse_args()
    records = [inspect(name, m, budget) for name in ("DPF3", "DPF5")
               for m in args.objectives for budget in args.budgets]
    print(json.dumps({
        "schema": "gt119_dpf_branch_finite_sample_v1",
        "limitation": "diagnostic only; no global PF membership or completeness established",
        "records": records,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
