# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent objective-space DTLZ8 Pareto membership diagnostics.

An internally nondominated sample need not lie on the actual front.
The test here reconstructs the published line and patch equations,
without consulting the native PS sampler or static reference files.
"""
import json
import argparse

import numpy as np
from moeabench.mops.DTLZ8 import DTLZ8
from scripts.audit.gt_dtlz8_candidate_issue119 import candidate_objectives


def membership_residual(objectives):
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    m = f.shape[1]
    if m < 2:
        raise ValueError("Expected M>=2.")
    t = f[:, -1]
    y = f[:, :-1]
    residual = np.empty(len(f), dtype=float)

    if m == 2:
        residual[:] = np.abs(y[:, 0] - (1-t)/4)
    else:
        line = t >= 1/3
        if np.any(line):
            residual[line] = np.max(
                np.abs(y[line] - ((1-t[line])/4)[:, None]), axis=1)
        if np.any(~line):
            z = np.sort(y[~line], axis=1)
            b = 1 - 2*t[~line]
            a = (1-t[~line])/4
            pair_error = np.abs(z[:, 0]+z[:, 1]-b)
            minimum_error = np.maximum(a-z[:, 0], 0)
            upper_equal_error = (
                np.max(np.abs(z[:, 2:]-z[:, 1:2]), axis=1)
                if m > 3 else np.zeros(np.count_nonzero(~line)))
            residual[~line] = np.maximum.reduce(
                (pair_error, minimum_error, upper_equal_error))
    return residual


def feasibility_residual(objectives):
    """Independent maximum violation of the published inequalities."""
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    m = f.shape[1]
    t, y = f[:, -1], f[:, :-1]
    violations = np.maximum(1 - t[:, None] - 4*y, 0).max(axis=1)
    if m >= 3:
        sorted_y = np.sort(y, axis=1)
        pair_violation = np.maximum(1 - 2*t - sorted_y[:, 0]
                                     - sorted_y[:, 1], 0)
        violations = np.maximum(violations, pair_violation)
    return violations


def audit(m, budgets=(128, 512, 2000), tolerance=1e-10):
    mop = DTLZ8(M=m)
    cases = []
    for budget in budgets:
        incumbent = np.asarray(mop.pf(n_points=budget), dtype=float)
        halton = candidate_objectives(m, budget)
        lattice = candidate_objectives(m, budget, patch_strategy="lattice")
        results = {}
        for name, f in [
            ("incumbent", incumbent), ("candidate_halton", halton),
            ("candidate_lattice", lattice),
        ]:
            r = membership_residual(f)
            g = feasibility_residual(f)
            results[name] = {
                "count": len(f),
                "unique": len(np.unique(f, axis=0)),
                "max_residual": float(np.max(r)),
                "max_constraint_violation": float(np.max(g)),
                "constraint_violations_above_tol": int(np.count_nonzero(g > tolerance)),
                "outside_tol": int(np.count_nonzero(r > tolerance)),
                "tolerance": tolerance,
            }
        cases.append({"M": m, "requested": budget, "results": results})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objectives", type=int, nargs="+",
                        default=[2, 3, 4, 5, 10])
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[128, 512, 2000])
    args = parser.parse_args()
    records = [record for m in args.objectives
               for record in audit(m, args.budgets)]
    print(json.dumps({"schema": "dtlz8_independent_membership_issue119_v1",
                      "records": records}, indent=2))


if __name__ == "__main__":
    main()
