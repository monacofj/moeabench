# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Provisional paired GT evidence for original DTLZ5/6 (ADR 0059).

The existing calibration sidecar stores objectives only, which cannot
reconstruct the *particular generating decisions*. This opt-in research
tool stores a separately identified (PS, F) pair, checks original
decision-box feasibility and an INDEPENDENT implementation of the
published objective equations, and optionally compares the ordered
objective rows with a calibration sidecar.

It never overwrites an existing artifact, baseline or calibration
sidecar, changes a sampler, or promotes approximate PF membership to
scientific GT certification. For source/platform provenance, capture a
separate gt_evidence_manifest.py manifest over the generated .npz file.

CLI (a full 2,000-point generation may be expensive; run locally only
after selecting a source commit, environment and experiment parameters):
    python scripts/audit/dtlz56_paired_gt_evidence.py sample \\
      --problem DTLZ5 --m 4 --k 4 --points 2000 \\
      --bundle dtlz5-M4-paired.npz
    python scripts/audit/dtlz56_paired_gt_evidence.py verify \\
      --problem DTLZ5 --m 4 --k 4 --bundle dtlz5-M4-paired.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import pi
from pathlib import Path

import numpy as np


SCHEMA = "moeabench_dtlz56_paired_gt_candidate_v1"


def _original_objectives(x, *, problem, m, k):
    """Independent published equations, not the production MOP evaluator."""
    if problem not in ("DTLZ5", "DTLZ6") or type(m) is not int or m < 2:
        raise ValueError("Require original DTLZ5/6 and integer M>=2")
    if type(k) is not int or k < 1:
        raise ValueError("Require integer K>=1")
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != m + k - 1:
        raise ValueError("Decision matrix shape must be (rows, M+K-1)")
    if x.shape[0] < 1 or not np.all(np.isfinite(x)):
        raise ValueError("Require a nonempty finite decision matrix")
    if np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError("Original DTLZ5/6 decisions must lie in [0,1]")

    tail = x[:, m-1:]
    g = (np.sum((tail - 0.5)**2, axis=1) if problem == "DTLZ5"
         else np.sum(tail**0.1, axis=1))
    r = 1.0 + g
    angles = np.empty((len(x), m-1), dtype=float)
    angles[:, 0] = x[:, 0] * (pi / 2.0)
    angles[:, 1:] = (
        (pi/(4*r[:, None])) * (1+2*g[:, None]*x[:, 1:m-1])
    )
    y = np.empty((len(x), m), dtype=float)
    for j in range(m):
        values = r.copy()
        for index in range(m-j-1):
            values *= np.cos(angles[:, index])
        if j:
            values *= np.sin(angles[:, m-j-1])
        y[:, j] = values
    return y, g


def audit_paired_candidate(
    decisions, objectives, *, problem, m, k, tolerance=1e-9,
    sidecar=None,
):
    """Independently check decision feasibility, row alignment and sidecar.

    This is a read-only, *per-row feasibility* audit. Even if all
    numerical checks pass, exact global Pareto membership, finite
    reference quality and environment/result gates remain UNVERIFIED.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Require finite strictly positive tolerance")
    x = np.asarray(decisions, dtype=float)
    f = np.asarray(objectives, dtype=float)
    expected, g = _original_objectives(x, problem=problem, m=m, k=k)
    if f.shape != expected.shape or not np.all(np.isfinite(f)):
        raise ValueError("Objectives must match finite decision-row shape")
    residual = np.max(np.abs(f-expected), axis=1)
    max_error = float(np.max(residual))
    matched = bool(np.all(residual <= tolerance))
    sidecar_status = "not_requested"
    sidecar_error = None
    if sidecar is not None:
        payload = json.loads(Path(sidecar).read_text(encoding="utf-8"))
        if (payload.get("problem_id") != problem or
                int(payload.get("mop_dimension", -1)) != m):
            sidecar_status = "mismatched_metadata"
        else:
            reference = np.asarray(payload.get("gt_reference"), dtype=float)
            if reference.shape != f.shape or not np.all(np.isfinite(reference)):
                sidecar_status = "mismatched_shape_or_nonfinite"
            else:
                sidecar_error = float(np.max(np.abs(reference-f)))
                sidecar_status = (
                    "numerically_aligned" if sidecar_error <= tolerance
                    else "mismatched_objective_rows"
                )
    return {
        "schema": SCHEMA,
        "problem": problem, "M": m, "K": k,
        "n_points": len(x),
        "tolerance": float(tolerance),
        "decision_feasibility": "pass",
        "independent_objective_alignment": "pass" if matched else "fail",
        "misaligned_rows": int(np.count_nonzero(residual > tolerance)),
        "max_abs_objective_error": max_error,
        "positive_g_rows": int(np.count_nonzero(g > 0.0)),
        "zero_g_rows": int(np.count_nonzero(g == 0.0)),
        "sidecar_alignment": sidecar_status,
        "sidecar_max_abs_error": sidecar_error,
        "global_pareto_membership": "unverified",
        "finite_gt_coverage": "unverified",
        "scientific_certification": "unverified",
    }


def _bundle_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def save_paired_candidate(path, decisions, objectives, *, problem, m, k,
                          tolerance=1e-9, sidecar=None):
    """Write once: never replace an artifact or treat failures as certified."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Paired evidence already exists: {destination}")
    report = audit_paired_candidate(
        decisions, objectives, problem=problem, m=m, k=k,
        tolerance=tolerance, sidecar=sidecar)
    if report["independent_objective_alignment"] != "pass":
        raise ValueError("Refusing to persist misaligned decision/objective rows")
    with destination.open("xb") as output:
        np.savez_compressed(
            output, decisions=np.asarray(decisions, dtype=float),
            objectives=np.asarray(objectives, dtype=float),
            problem=np.asarray(problem), M=np.asarray(m), K=np.asarray(k),
        )
    return report | {"artifact_sha256": _bundle_digest(destination)}


def verify_paired_candidate(path, *, problem, m, k, tolerance=1e-9,
                            sidecar=None, expected_sha256=None):
    """Recompute the independent equations from the saved aligned pair."""
    bundle = Path(path)
    digest = _bundle_digest(bundle)
    with np.load(bundle, allow_pickle=False) as data:
        if set(data.files) != {"decisions", "objectives", "problem", "M", "K"}:
            raise ValueError("Unexpected paired GT bundle members")
        if (str(data["problem"].item()) != problem or
                int(data["M"].item()) != m or int(data["K"].item()) != k):
            raise ValueError("Bundle problem/dimension metadata mismatch")
        result = audit_paired_candidate(
            data["decisions"], data["objectives"], problem=problem,
            m=m, k=k, tolerance=tolerance, sidecar=sidecar)
    result["artifact_sha256"] = digest
    result["expected_digest_match"] = (
        None if expected_sha256 is None else digest == expected_sha256)
    return result


def sample_candidate(*, problem, m, k, points, bundle, tolerance=1e-9):
    """Explicitly expensive candidate generation, never automatic."""
    if type(points) is not int or points < 1:
        raise ValueError("points must be a positive integer")
    if Path(bundle).exists():
        raise FileExistsError(f"Refusing to overwrite {bundle}")
    from moeabench.mops import DTLZ5, DTLZ6
    mop = (DTLZ5 if problem == "DTLZ5" else DTLZ6)(
        M=m, K=k)
    # The sampler is invoked ONCE; preserving PS/F alignment is the
    # purpose of this external research artifact.
    ps = mop.ps(n_points=points)
    objective_rows = mop.evaluation(ps)["F"]
    return save_paired_candidate(
        bundle, ps, objective_rows, problem=problem, m=m, k=k,
        tolerance=tolerance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sample", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--problem", required=True,
                             choices=("DTLZ5", "DTLZ6"))
        command.add_argument("--m", type=int, required=True)
        command.add_argument("--k", type=int, required=True)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--tolerance", type=float, default=1e-9)
        if name == "sample":
            command.add_argument("--points", type=int, required=True)
        else:
            command.add_argument("--sidecar", type=Path)
            command.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    if args.command == "sample":
        result = sample_candidate(
            problem=args.problem, m=args.m, k=args.k, points=args.points,
            bundle=args.bundle, tolerance=args.tolerance)
    else:
        result = verify_paired_candidate(
            args.bundle, problem=args.problem, m=args.m, k=args.k,
            tolerance=args.tolerance, sidecar=args.sidecar,
            expected_sha256=args.expected_sha256)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if (result["independent_objective_alignment"] != "pass" or
            result.get("expected_digest_match") is False or
            result["sidecar_alignment"] not in (
                "not_requested", "numerically_aligned")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
