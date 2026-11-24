# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small source-bound pilot of the actual DTLZ5/6 M>=4 scalarization path.

Runs the production scalarized_candidate + decision_vector pipeline with a
small, FIXED set of prescribed preference rays, and compares its results
to independent original equations and conditional symbolic theorem reach.
This deliberately does NOT call .ps(), which would perform >=448 scalarized
solves even for a small nominal sample. It does not certify the full GT.

Example (CPU-only; no files except optional --output):
  python -m scripts.audit.dtlz56_small_scalarization_pilot \\
    --problem DTLZ5 --m 4 --k 1 --output dtlz5-M4-pilot.json

Scientific safeguards: the experiment reports raw optimizer gap, actual
radial g of recovered decisions, independent row alignment, and all three
conditional proof statuses. A small pilot is NOT an adequate front sample.
Reference environment / ADR0059 certification remains unverified.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from moeabench.mops import DTLZ5, DTLZ6
from moeabench.mops._dtlz56_front import (
    scalarized_candidate, decision_vector,
)
from .dtlz56_paired_gt_evidence import _original_objectives
from .dtlz56_paired_symbolic_audit import audit_symbolic_candidate


SCHEMA = "moeabench_dtlz56_small_scalarization_pilot_v1"

# Preference rays adapted from exact original feasible *boundary* positions,
# with no RNG. This samples no implied measure on the full front.
RAYS = (
    (0.0625, 1.0, 0.0, 0.1),
    (0.25, 1.0, 0.0, 0.2),
    (0.125, 0.0, 1.0, 0.1),
    (0.5, 0.0, 1.0, 0.2),
)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except (OSError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return "unavailable"


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def run_pilot(*, problem, m=4, k=1, abs_tol=1e-5, max_nodes=5000):
    """Run bounded production scalarized solves; no PS/pf generation."""
    if problem not in ("DTLZ5", "DTLZ6") or type(m) is not int or m != 4:
        raise ValueError("Pilot currently supports original DTLZ5/6 M=4 only")
    if type(k) is not int or not 1 <= k <= 8:
        raise ValueError("Expected integer K from 1 to 8")
    if not np.isfinite(abs_tol) or not 0 < abs_tol <= 1e-4:
        raise ValueError("Expected 0 < abs_tol <= 1e-4")
    if type(max_nodes) is not int or not 1 <= max_nodes <= 12000:
        raise ValueError("Expected integer max_nodes from 1 to 12000")

    mop = (DTLZ5 if problem == "DTLZ5" else DTLZ6)(M=m, K=k)
    gmax = k/4 if problem == "DTLZ5" else float(k)
    input_decisions = np.empty((len(RAYS), m+k-1), dtype=float)
    for i,(first,second,third,g) in enumerate(RAYS):
        if g >= gmax:
            raise ValueError("Pilot ray g must lie strictly below original maximum")
        input_decisions[i,:m-1] = (first,second,third)
        if problem == "DTLZ5":
            input_decisions[i,m-1:] = .5+np.sqrt(g/k)
        else:
            input_decisions[i,m-1:] = (g/k)**10

    # Generate source rays with the original MOP evaluator just as the
    # production sampler does; use its actual scalarized optimizer.
    preference_rays = mop.evaluation(input_decisions)["F"]
    preference_rays = np.maximum(
        preference_rays,
        1.01e-7*np.max(preference_rays,axis=1,keepdims=True),
    )
    candidate_rows = []
    ps = []
    for index,weights in enumerate(preference_rays):
        candidate = scalarized_candidate(
            weights, gmax, abs_tol=abs_tol, max_nodes=max_nodes,
        )
        decision = decision_vector(candidate,M=m,K=k,problem=problem)
        candidate_rows.append({
            "ray_index": index,
            "source_ray": list(RAYS[index]),
            "optimizer_radial_g": float(candidate.g),
            "scalarization_abs_gap": float(candidate.abs_gap),
            "scalarization_lower_bound": float(candidate.lower_bound),
            "scalarization_scale": float(candidate.scale),
            "nodes": int(candidate.nodes),
        })
        ps.append(decision)

    decisions = np.asarray(ps, dtype=float)
    objective_rows = mop.evaluation(decisions)["F"]
    independent, independent_g = _original_objectives(
        decisions, problem=problem, m=m, k=k,
    )
    alignment = float(np.max(np.abs(objective_rows-independent)))
    audit = audit_symbolic_candidate(
        decisions, objective_rows, problem=problem, m=m, k=k,
    )
    for entry,candidate in zip(audit["rows"],candidate_rows):
        entry["ray_index"] = candidate["ray_index"]
        entry["evaluated_radial_g"] = float(independent_g[candidate["ray_index"]])
    if audit["independent_objective_alignment"] != "pass":
        raise AssertionError("Production scalarization decisions failed independent evaluation")

    # Separate opt-in theorem ledger: does not change the established
    # exact-rational-g classifier contract or any stored GT status.
    irrational_dtlz6_support = None
    if problem == "DTLZ6":
        from .dtlz56_dtlz6_radial_enclosure import (
            audit_dtlz6_irrational_radial_rows,
        )
        irrational_dtlz6_support = audit_dtlz6_irrational_radial_rows(
            decisions, objective_rows, m=m, k=k,
        )

    # Complement the earlier *membership* ledger with conditional
    # distances to independently proved PF points. Never promote it to
    # membership; preserve individual exact source decision coordinates.
    support_proximity = None
    if problem == "DTLZ6":
        from .dtlz56_dtlz6_support_proximity import (
            certify_dtlz6_support_proximity,
        )
        support_proximity = []
        for i, decision in enumerate(decisions):
            cert = certify_dtlz6_support_proximity(
                decision[:m-1], decision[m-1:], k=k,
            )
            support_proximity.append({
                "row": i,
                "status": cert.conclusion,
                "certificate": cert.to_dict(),
            })

    result = {
        "schema": SCHEMA,
        "exact_source_decisions": decisions.tolist(),
        "dtlz6_certified_support_proximity": support_proximity,
        "dtlz6_irrational_radial_support": irrational_dtlz6_support,
        "problem": problem,
        "M": m, "K": k,
        "method": "production_scalarized_candidate_and_decision_vector",
        "scope": "fixed_preference_rays_not_full_gt_or_front_cover",
        "ray_count": len(RAYS),
        "abs_tol": float(abs_tol),
        "max_nodes": max_nodes,
        "fixed_rays": [list(v) for v in RAYS],
        "source_sha": _git_commit(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "decision_bytes_sha256": sha256(decisions.tobytes()).hexdigest(),
        "objective_bytes_sha256": sha256(objective_rows.tobytes()).hexdigest(),
        "independent_max_abs_objective_error": alignment,
        "optimizer": candidate_rows,
        "per_row_audit": audit,
        "scientific_certification": "unverified",
        "full_gt_coverage": "unverified",
    }
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem",choices=("DTLZ5","DTLZ6"),required=True)
    parser.add_argument("--m",type=int,default=4)
    parser.add_argument("--k",type=int,default=1)
    parser.add_argument("--abs-tol",type=float,default=1e-5)
    parser.add_argument("--max-nodes",type=int,default=5000)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    report=run_pilot(
        problem=args.problem,m=args.m,k=args.k,
        abs_tol=args.abs_tol,max_nodes=args.max_nodes,
    )
    payload=json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x",encoding="utf-8") as output:
            output.write(payload)
    else:
        print(payload,end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
