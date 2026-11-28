# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded sensitivity experiment for source Sobol ray 157 (DTLZ5 M4K1).

Not a repair, GT sampler modification, or exact Pareto membership claim.
Recompute the SAME original source preference ray and run isolated
unmodified scalarizer tolerances, without full PS or maximin selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from moeabench.mops import DTLZ5
from moeabench.mops._dtlz56_front import (
    scalarized_candidate, decision_vector,
)
from .dtlz56_paired_gt_evidence import _original_objectives
from .dtlz5_row10_exact_dominance import SOURCE_ROW_10

INDEX = 157
SOURCE_WEIGHT_VALUES = (
    0.4789323296283287, 0.5159827838064277,
    0.7300843104029774, 0.2708136316666282,
)


def original_source_ray_157():
    """Reproduce the original source-bound Sobol-weight construction."""
    mop = DTLZ5(M=4, K=1)
    sampled = qmc.Sobol(d=4, scramble=True, seed=46).random_base2(8)[INDEX]
    position = np.zeros((1, 4), dtype=float)
    position[0, 0] = sampled[0]
    position[0, 1:3] = sampled[2:4]
    position[0, 1 + INDEX % 2] = INDEX % 2
    g = 0.25 * sampled[1]**2
    position[0, 3] = 0.5 + np.sqrt(g)
    raw = np.maximum(mop.evaluation(position)["F"][0], 1e-12)
    weights = np.maximum(raw, 1.01e-7 * np.max(raw))
    if not np.array_equal(weights, np.asarray(SOURCE_WEIGHT_VALUES)):
        raise AssertionError("Sobol source preference differs from archived source")
    return weights


def sweep_source_ray_157(*, tolerances=(1e-7, 1e-9, 1e-11)):
    weights = original_source_ray_157()
    observations = []
    for tol in tolerances:
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("Positive finite tolerances required")
        try:
            candidate = scalarized_candidate(
                weights, 0.25, abs_tol=tol, max_nodes=30000,
            )
            decision = decision_vector(
                candidate, M=4, K=1, problem="DTLZ5",
            )
            objectives, _ = _original_objectives(
                decision[None, :], problem="DTLZ5", m=4, k=1,
            )
            observations.append({
                "tolerance": float(tol),
                "status": "numerically_completed",
                "nodes": int(candidate.nodes),
                "achieved_scalarization_gap_bound": float(candidate.abs_gap),
                "g": float(candidate.g),
                "decision_hex": [float(v).hex() for v in decision],
                "solver_objectives": candidate.objectives.tolist(),
                "independent_objectives": objectives[0].tolist(),
                "exact_archived_row10_match": bool(
                    tuple(float(v) for v in decision) == SOURCE_ROW_10
                ),
                "scientific_membership": "unverified",
            })
        except RuntimeError as exc:
            observations.append({
                "tolerance": float(tol), "status": "budget_exhausted",
                "reason": str(exc), "scientific_membership": "unverified",
            })
    return {
        "schema": "dtlz5_candidate157_tolerance_sweep_v1",
        "source_candidate": INDEX,
        "source_weights": weights.tolist(),
        "source": "independent_original_Sobol_256_seed46",
        "observations": observations,
        "scientific_certification": "unverified",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Refusing to overwrite source research report")
    result = sweep_source_ray_157()
    with args.report.open("x", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True, allow_nan=False)
        fp.write("\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
