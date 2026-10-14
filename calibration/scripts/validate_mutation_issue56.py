# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare legacy and corrected pymoo mutation semantics on many-objective DTLZ3.

This is a scientific validation helper for issue #56 / PR #57, not a
regression test.  DTLZ3 has a unit-hypersphere Pareto front, so for every
objective vector F the radial error ||F|| - 1 is exactly the distance factor
introduced by the DTLZ3 g function.  This gives a direct convergence signal
without relying on high-dimensional hypervolume estimation.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from contextlib import contextmanager

import numpy as np
from pymoo.operators.mutation.pm import PolynomialMutation


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import moeabench as mb
from moeabench.moeas._operators_pymoo import standard_polynomial_mutation


ALGORITHMS = {
    "NSGA3": ("moeabench.moeas._nsga_pymoo", "NSGA3"),
    "U_NSGA3": ("moeabench.moeas._unsga_pymoo", "U_NSGA3"),
    "MOEAD": ("moeabench.moeas._moead_pymoo", "MOEAD"),
    "RVEA": ("moeabench.moeas._rvea_pymoo", "RVEA"),
    "SPEA2": ("moeabench.moeas._spea_pymoo", "SPEA2"),
}


def legacy_polynomial_mutation(n_var, eta=20):
    """Reproduce MoeaBench's pre-fix call under pymoo >= 0.6 semantics."""
    return PolynomialMutation(prob=1.0 / n_var, eta=eta)


@contextmanager
def mutation_policy(module_name, policy):
    """Temporarily replace the shared mutation factory imported by one wrapper."""
    module = importlib.import_module(module_name)
    original = module.standard_polynomial_mutation
    module.standard_polynomial_mutation = policy
    try:
        yield
    finally:
        module.standard_polynomial_mutation = original


def radial_summary(front):
    """Summarize convergence to DTLZ3's unit-hypersphere Pareto front."""
    front = np.asarray(front, dtype=float)
    radial_error = np.linalg.norm(front, axis=1) - 1.0
    return {
        "n": len(radial_error),
        "best": float(np.min(radial_error)),
        "median": float(np.median(radial_error)),
        "mean": float(np.mean(radial_error)),
        "p90": float(np.quantile(radial_error, 0.90)),
        "within_1e-1": float(np.mean(radial_error <= 1e-1)),
        "within_1e-2": float(np.mean(radial_error <= 1e-2)),
    }


def run_case(algorithm_name, policy_name, policy, objectives, population, generations, seed):
    module_name, public_name = ALGORITHMS[algorithm_name]
    algorithm_cls = getattr(mb.moeas, public_name)

    mop = mb.mops.DTLZ3(M=objectives)
    moea = algorithm_cls(
        population=population,
        generations=generations,
        seed=seed,
    )
    exp = mb.experiment(mop=mop, moea=moea)

    with mutation_policy(module_name, policy):
        exp.run(repeat=1, silent=True)

    result = radial_summary(exp.last_run.front())
    result.update(
        algorithm=algorithm_name,
        policy=policy_name,
        seed=seed,
    )
    return result


def print_results(rows):
    headers = (
        "algorithm",
        "policy",
        "seed",
        "n",
        "best",
        "median",
        "mean",
        "p90",
        "within_1e-1",
        "within_1e-2",
    )
    print("\t".join(headers))
    for row in rows:
        fields = []
        for key in headers:
            value = row[key]
            if isinstance(value, float):
                fields.append(f"{value:.6g}")
            else:
                fields.append(str(value))
        print("\t".join(fields))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the pre-fix and corrected polynomial mutation semantics "
            "on many-objective DTLZ3."
        )
    )
    parser.add_argument(
        "--algorithm",
        choices=tuple(ALGORITHMS),
        action="append",
        default=None,
        help="Algorithm to validate; may be repeated. Default: NSGA3.",
    )
    parser.add_argument("--objectives", type=int, default=15)
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Execution seed; may be repeated. Default: 12345.",
    )
    args = parser.parse_args()

    algorithms = args.algorithm or ["NSGA3"]
    seeds = args.seed or [12345]

    rows = []
    for algorithm_name in algorithms:
        for seed in seeds:
            print(
                f"Running {algorithm_name}, legacy mutation, seed={seed}...",
                flush=True,
            )
            rows.append(
                run_case(
                    algorithm_name,
                    "legacy",
                    legacy_polynomial_mutation,
                    args.objectives,
                    args.population,
                    args.generations,
                    seed,
                )
            )

            print(
                f"Running {algorithm_name}, corrected mutation, seed={seed}...",
                flush=True,
            )
            rows.append(
                run_case(
                    algorithm_name,
                    "corrected",
                    standard_polynomial_mutation,
                    args.objectives,
                    args.population,
                    args.generations,
                    seed,
                )
            )

    print()
    print_results(rows)


if __name__ == "__main__":
    main()
