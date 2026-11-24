# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only, conservative per-row symbolic audit of paired DTLZ5/6 candidates.

Rational decision coordinates from an NPZ (IEEE-754 doubles) are treated
as their EXACT binary rationals, not rounded decimal literals. The radial
distance g is computed exactly for DTLZ5. For DTLZ6, rationality of EVERY
x**(1/10) tail term is proved using perfect integer tenth-power checks;
otherwise that row is unverified, never assigned an approximate g.

For supported rows, mathematical statements concern the exact-real
original objective image of that *decision*, CONDITIONAL on its saved
finite-precision objective row being aligned. Numerical alignment, even
when below tolerance, is not an exact identity proof for stored F.
No result promotes whole-GT membership, finite coverage, numerical
reproducibility or ADR 0059 acceptance gates to pass.

Usage (never generates GT):
  python -m scripts.audit.dtlz56_paired_symbolic_audit \\
      --bundle input-paired.npz --problem DTLZ5 --m 4 --k 4
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from .dtlz56_paired_gt_evidence import (
    _original_objectives,
    audit_paired_candidate,
)
from .dtlz56_exact_same_ray import (
    construct_same_ray_witness,
)
from .dtlz56_generalized_positive_g_pf import (
    certify_generalized_positive_g_pf,
)
from .dtlz56_positive_g_open_first_angle import (
    certify_open_first_angle_global_pf,
)


SCHEMA = "moeabench_dtlz56_paired_symbolic_audit_v1"


def _integer_root_exact(value: int, power: int):
    """Return exact nonnegative integer root or None (no floating point)."""
    if value < 0:
        raise ValueError("nonnegative integer required")
    if value < 2:
        return value
    lo, hi = 0, 1 << ((value.bit_length() + power - 1)//power)
    while lo <= hi:
        mid = (lo+hi)//2
        raised = mid**power
        if raised == value:
            return mid
        if raised < value:
            lo = mid+1
        else:
            hi = mid-1
    return None


def _exact_tenth_root(value: Fraction):
    """Prove rational tenth root, or decline; no numeric approximation."""
    if value < 0:
        raise ValueError("DTLZ6 tail must be nonnegative")
    numerator = _integer_root_exact(value.numerator, 10)
    denominator = _integer_root_exact(value.denominator, 10)
    if numerator is None or denominator is None:
        return None
    return Fraction(numerator, denominator)


def _exact_radial_g(decision, *, problem, m):
    tail = [Fraction.from_float(float(v)) for v in decision[m-1:]]
    if problem == "DTLZ5":
        return sum(((v-Fraction(1, 2))**2 for v in tail), Fraction(0))
    roots = [_exact_tenth_root(v) for v in tail]
    if any(root is None for root in roots):
        return None
    return sum(roots, Fraction(0))


def _row_symbolic_result(decision, *, problem, m, k):
    g = _exact_radial_g(decision, problem=problem, m=m)
    if g is None:
        return {
            "status": "unverified",
            "basis": "Original DTLZ6 radial g not exactly rational from stored tail",
            "g_exact": None,
            "theorem": None,
        }
    positions = tuple(Fraction.from_float(float(v)) for v in decision[:m-1])
    exact_g = {
        "numerator_hex": hex(g.numerator),
        "denominator_hex": hex(g.denominator),
    }
    if g == 0:
        return {
            "status": "symbolic_decision_global_pf",
            "basis": "Unit-radius global norm theorem for original DTLZ5/6",
            "g_exact": exact_g,
            "theorem": "DTLZ56-UNIT-RADIUS",
        }
    if m <= 3:
        return {
            "status": "unverified",
            "basis": "M<=3 positive-g proof is not implemented in this research classifier",
            "g_exact": exact_g,
            "theorem": None,
        }
    same_ray = construct_same_ray_witness(
        positions, g=g, problem=problem, k=k,
    )
    if same_ray.conclusion == "exact_same_ray_dominator":
        return {
            "status": "symbolic_decision_dominated",
            "basis": "Exact rational strict same-ray smaller-g witness",
            "g_exact": exact_g,
            "theorem": "DTLZ56-SAME-RAY",
        }
    global_boundary = certify_generalized_positive_g_pf(
        positions, g=g, problem=problem, k=k,
    )
    if global_boundary.conclusion == "rigorously_global_pf_member":
        return {
            "status": "symbolic_decision_global_pf",
            "basis": "Exact global sine-power supporting stratum",
            "g_exact": exact_g,
            "theorem": global_boundary.proof_id,
        }
    global_open = certify_open_first_angle_global_pf(
        positions, g=g, problem=problem, k=k,
    )
    if global_open.conclusion == "rigorously_global_pf_member":
        return {
            "status": "symbolic_decision_global_pf",
            "basis": "Exact-rational sufficient open-first-angle global theorem",
            "g_exact": exact_g,
            "theorem": global_open.proof_id,
        }
    return {
        "status": "unverified",
        "basis": "No applicable exact symbolic sufficient membership/dominance theorem",
        "g_exact": exact_g,
        "theorem": None,
    }


def audit_symbolic_candidate(
    decisions, objectives, *, problem, m, k, tolerance=1e-9,
):
    """Classify underlying exact decisions; conservatively gate bad alignment."""
    prior = audit_paired_candidate(
        decisions, objectives, problem=problem, m=m, k=k,
        tolerance=tolerance,
    )
    x = np.asarray(decisions, dtype=float)
    expected, _ = _original_objectives(x, problem=problem, m=m, k=k)
    supplied = np.asarray(objectives, dtype=float)
    residual = np.max(np.abs(expected-supplied), axis=1)
    records = []
    for row, decision in enumerate(x):
        if residual[row] > tolerance:
            entry = {
                "status": "unverified",
                "basis": "Saved objective row does not align with original decision",
                "g_exact": None,
                "theorem": None,
            }
        else:
            entry = _row_symbolic_result(
                decision, problem=problem, m=m, k=k,
            )
        records.append({"row": row, **entry})
    counts = Counter(entry["status"] for entry in records)
    return {
        "schema": SCHEMA,
        "problem": problem, "M": m, "K": k,
        "n_points": len(x),
        "independent_objective_alignment":
            prior["independent_objective_alignment"],
        "counts": {
            key: counts.get(key, 0)
            for key in (
                "symbolic_decision_global_pf",
                "symbolic_decision_dominated",
                "unverified",
            )
        },
        "rows": records,
        "whole_reference_membership": "unverified",
        "finite_gt_coverage": "unverified",
        "scientific_certification": "unverified",
        "interpretation": (
            "Symbolic claims apply to the exact-real image of stored "
            "decision coordinates; finite-precision objective alignment "
            "is only checked numerically, not proved as exact equality."
        ),
    }


def audit_paired_bundle(path, *, problem, m, k, tolerance=1e-9,
                        expected_sha256=None):
    """Only reads an existing paired bundle; never runs a PS sampler."""
    bundle = Path(path)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Paired artifact digest mismatch")
    with np.load(bundle, allow_pickle=False) as data:
        if set(data.files) != {"decisions", "objectives", "problem", "M", "K"}:
            raise ValueError("Unexpected paired GT bundle members")
        if (data["problem"].item() != problem or
                int(data["M"].item()) != m or int(data["K"].item()) != k):
            raise ValueError("Paired bundle original problem or dimensions mismatch")
        output = audit_symbolic_candidate(
            data["decisions"], data["objectives"], problem=problem,
            m=m, k=k, tolerance=tolerance,
        )
    output["artifact_sha256"] = digest
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--problem", choices=("DTLZ5","DTLZ6"), required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    output = audit_paired_bundle(
        args.bundle, problem=args.problem, m=args.m, k=args.k,
        tolerance=args.tolerance, expected_sha256=args.expected_sha256,
    )
    print(json.dumps(output, sort_keys=True, indent=2, allow_nan=False))
    return 0 if output["independent_objective_alignment"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
