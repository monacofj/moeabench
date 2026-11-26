# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Research-only proof-gated original DTLZ5 source candidate refinement.

This prototype handles one fixed source preference ray, independently checks
exact-real decision-image membership, and accepts ONLY mathematically proved
members. It deliberately abstains if no proof is found; it cannot assert that
every true PF point matches one of the presently implemented theorems.

No changes to the production sampler, its objectives, its historical tests,
or ADR 0059. A finite scalar tolerance is never a PF certificate.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from moeabench.mops._dtlz56_front import scalarized_candidate, decision_vector

from .dtlz5_candidate157_tolerance_sweep import original_source_ray_157
from .dtlz56_tight_sine_global_pf import (
    certify_tight_sine_global_pf, replay_tight_sine_global_pf,
)
from .dtlz56_exact_same_ray import construct_same_ray_witness
from .dtlz5_row10_exact_dominance import (
    SOURCE_ROW_10, archived_row10_witness, certify_strict_dtlz5_m4_dominator,
)

SCHEMA = "dtlz5_source157_proof_gated_refinement_v1"


def select_proven_source_candidate(
    weights, *, tolerances=(1e-7, 1e-9, 1e-11),
    solver=scalarized_candidate, reconstruct=decision_vector,
):
    """Return first EXACT positive Pareto proof, or fail closed.

    The solver and reconstructor are injected only to permit lightweight
    tests; the CLI uses the actual original production helpers.
    """
    attempts = []
    selected = None
    for tolerance in tolerances:
        if not isinstance(tolerance, (float, int)) or not 0 < tolerance < 1:
            raise ValueError("Expected finite scalar tolerance in (0,1)")
        candidate = solver(weights, 0.25, abs_tol=tolerance, max_nodes=30000)
        decision = tuple(float(v) for v in reconstruct(
            candidate, M=4, K=1, problem="DTLZ5",
        ))
        if len(decision) != 4 or any(not 0 <= x <= 1 for x in decision):
            raise ValueError("Solver returned infeasible original decision")
        g = (Fraction.from_float(decision[3])-Fraction(1, 2))**2
        proof = certify_tight_sine_global_pf(
            decision[:3], g=g, problem="DTLZ5", k=1,
        )
        accepted = (
            proof.conclusion == "rigorously_global_pf_member"
            and replay_tight_sine_global_pf(
                decision[:3], proof.to_dict(), g=g, problem="DTLZ5", k=1,
            )
        )
        old_witness = None
        if decision == SOURCE_ROW_10:
            strict = certify_strict_dtlz5_m4_dominator(
                decision, archived_row10_witness(),
            )
            if strict["conclusion"] == "rigorously_strictly_dominated":
                old_witness = strict
        same_ray = construct_same_ray_witness(
            decision[:3], g=g, problem="DTLZ5", k=1,
        )
        dominated = (
            old_witness is not None or
            same_ray.conclusion == "exact_same_ray_dominator"
        )
        # Never promote a candidate with conflicting rigorous evidence.
        if accepted and dominated:
            raise AssertionError("Contradictory strict dominance and PF proofs")
        attempt = {
            "tolerance": float(tolerance),
            "decision_hex": [float(v).hex() for v in decision],
            "nodes": int(candidate.nodes),
            "solver_abs_gap": float(candidate.abs_gap),
            "exact_source_row10": decision == SOURCE_ROW_10,
            "proved_dominated": dominated,
            "exact_pf_proof": proof.to_dict() if accepted else None,
            "status": (
                "rigorously_global_decision_image" if accepted
                else "rigorously_dominated" if dominated else "unverified"
            ),
        }
        attempts.append(attempt)
        if accepted:
            selected = len(attempts)-1
            break

    return {
        "schema": SCHEMA,
        "problem": "DTLZ5", "M": 4, "K": 1,
        "attempts": attempts,
        "accepted_attempt_index": selected,
        "source_candidate_status": (
            "proved_global_decision_image" if selected is not None
            else "unverified_no_accepted_candidate"
        ),
        "whole_gt_membership": "unverified",
        "finite_gt_coverage": "unverified",
        "scientific_certification": "unverified",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Refusing to replace research proof output")
    result = select_proven_source_candidate(original_source_ray_157())
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "accepted_attempt_index": result["accepted_attempt_index"],
        "attempts": [
            {key: attempt[key] for key in (
                "tolerance", "status", "nodes", "solver_abs_gap",
                "exact_source_row10",
            )}
            for attempt in result["attempts"]
        ],
        "scientific_certification": "unverified",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
