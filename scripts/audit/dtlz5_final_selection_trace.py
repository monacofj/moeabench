# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in source-bound trace of original DTLZ5 final maximin selections.

Instrument *calls*, not production logic: for one unchanged public PS call
capture each scalarized_candidate and its reconstructed decision, then
compare solver objective vectors with independently reevaluated original
equations. All observations are numerical; exact Pareto membership is
decided only by separate rational certificates. No baseline is overwritten.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from moeabench.mops import DTLZ5
from moeabench.mops import _dtlz56_front as front
from .dtlz56_paired_gt_evidence import _original_objectives
from .dtlz5_row10_exact_dominance import SOURCE_ROW_10


def trace_original_dtlz5_selection(*, n_points=16, include_candidate_pool=False):
    if type(n_points) is not int or n_points != 16:
        raise ValueError("This research sentinel requires exactly 16 points")
    mop = DTLZ5(M=4, K=1)
    saved_solver = front.scalarized_candidate
    saved_reconstructor = front.decision_vector
    candidates = []
    reconstructions = []

    def observed_scalarizer(weights, max_g, **kwargs):
        candidate = saved_solver(weights, max_g, **kwargs)
        candidates.append({
            "weights": np.asarray(weights, dtype=float).copy(),
            "candidate": candidate,
        })
        return candidate

    def observed_reconstructor(candidate, **kwargs):
        position = np.asarray(
            saved_reconstructor(candidate, **kwargs), dtype=float
        ).copy()
        reconstructions.append(position)
        return position

    try:
        front.scalarized_candidate = observed_scalarizer
        front.decision_vector = observed_reconstructor
        selected = np.asarray(mop.ps(n_points=n_points), dtype=float)
    finally:
        front.scalarized_candidate = saved_solver
        front.decision_vector = saved_reconstructor

    if selected.shape != (16, 4):
        raise AssertionError("Unexpected original DTLZ5 selection shape")
    if len(candidates) != len(reconstructions):
        raise AssertionError("Scalarization/reconstruction call mismatch")
    if not candidates:
        raise AssertionError("No actual scalarizer calls were observed")

    all_x = np.asarray(reconstructions, dtype=float)
    all_y, _ = _original_objectives(
        all_x, problem="DTLZ5", m=4, k=1
    )
    solver_y = np.asarray(
        [record["candidate"].objectives for record in candidates],
        dtype=float,
    )
    deviations = all_y - solver_y
    abs_deviations = np.abs(deviations)
    rows = []
    for selected_index, row in enumerate(selected):
        matches = np.flatnonzero(
            np.all(all_x == row[None, :], axis=1)
        ).tolist()
        chosen_candidate = matches[0] if matches else None
        record = None
        if chosen_candidate is not None:
            item = candidates[chosen_candidate]
            candidate = item["candidate"]
            record = {
                "scalarizer_candidate_index": chosen_candidate,
                "matching_candidate_count": len(matches),
                "source_family": (
                    "sobol" if chosen_candidate < 256 else
                    "gaussian" if chosen_candidate < 384 else "gamma"
                ),
                "solver_objectives": solver_y[chosen_candidate].tolist(),
                "independent_reconstructed_objectives":
                    all_y[chosen_candidate].tolist(),
                "reconstruction_minus_solver_objectives":
                    deviations[chosen_candidate].tolist(),
                "reconstruction_max_abs_error":
                    float(np.max(abs_deviations[chosen_candidate])),
                "scalarized_g": float(candidate.g),
                "scalarized_scale": float(candidate.scale),
                "scalarized_abs_gap_bound": float(candidate.abs_gap),
                "scalarization_nodes": int(candidate.nodes),
                "active_faces": (
                    list(candidate.angular_active_faces)
                    if candidate.angular_active_faces is not None else None
                ),
                "source_weights": item["weights"].tolist(),
            }
        rows.append({
            "selected_index": selected_index,
            "decision_hex": [float(v).hex() for v in row],
            "is_exact_archived_row10": bool(
                tuple(float(v) for v in row) == SOURCE_ROW_10
            ),
            "trace": record,
        })
    archived = [
        r for r in rows if r["is_exact_archived_row10"]
    ]
    output = {
        "schema": "dtlz5_final_selection_source_diagnostics_v1",
        "problem": "DTLZ5", "M": 4, "K": 1,
        "points": n_points,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "num_scalarized_candidates": len(candidates),
        "max_absolute_objective_reconstruction_discrepancy":
            float(np.max(abs_deviations)),
        "selected_rows": rows,
        "archived_row10_exact_match_count": len(archived),
        "archived_row10_diagnostic": archived,
        "scientific_certification": "unverified",
    }
    if include_candidate_pool:
        # Explicit opt-in, lossless research observations; no change to PS.
        output["candidate_pool_decision_hex"] = [
            [float(v).hex() for v in row] for row in all_x
        ]
        output["candidate_pool_source_families"] = (
            ["sobol"]*256 + ["gaussian"]*128 + ["gamma"]*64
        )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Never overwrite an existing research report")
    result = trace_original_dtlz5_selection()
    with args.report.open("x", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True, allow_nan=False)
        fp.write("\n")
    print(json.dumps({
        "num_scalarized_candidates": result["num_scalarized_candidates"],
        "max_absolute_objective_reconstruction_discrepancy":
            result["max_absolute_objective_reconstruction_discrepancy"],
        "archived_row10_exact_match_count":
            result["archived_row10_exact_match_count"],
        "archived_row10_diagnostic": result["archived_row10_diagnostic"],
        "scientific_certification": "unverified",
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
