# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analytical decision-space preimages and repeated GT comparison, issue #119.

Read-only experiment. Runs unchanged DTLZ1-4 mathematical evaluation and
compares their unchanged ps()/pf() with independent objective-space FPS
candidates mapped back to genuine Pareto-set decision vectors.

CLI (no files written):
    python scripts/audit/gt_fps_preimage_study.py --objectives 3 5 \
        --budgets 128 512 --seeds 101 202 303 --candidate-pool 4096

Results are exploratory finite-reference diagnostics, never certified
continuous covering radii, calibrated references or acceptance thresholds.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from moeabench.mops.DTLZ2 import _hypersphere_directions_to_angles
from scripts.audit.gt_fps_comparison import farthest_candidates
from scripts.audit.gt_geometry_pilot import (
    _MOPS,
    geometry_stats,
    independent_reference,
)


def analytical_preimages(mop, front: np.ndarray) -> np.ndarray:
    """Invert a *valid* DTLZ1-4 analytical front to Pareto-set variables.

    Boundary preimages may be nonunique; choose zero for indeterminate
    decision-space coordinates. For every input point, require successful
    forward evaluation to verify the mapping rather than merely assuming
    membership from a geometry equation.
    """
    name = mop.__class__.__name__
    if name not in _MOPS:
        raise ValueError("Only exact DTLZ1-4 preimages are supported")
    front = np.asarray(front, dtype=np.float64)
    if (front.ndim != 2 or front.shape[1] != mop.M or len(front) == 0
            or not np.isfinite(front).all()):
        raise ValueError("Expected finite nonempty objective-space (n, M) points")
    target = 0.5 if name == "DTLZ1" else 1.0
    if np.any(front < -1e-13):
        raise ValueError("Objectives must lie on the positive Pareto manifold")
    geometry = front.sum(axis=1) if name == "DTLZ1" else np.linalg.norm(front, axis=1)
    if not np.allclose(geometry, target, rtol=0.0, atol=1e-11):
        raise ValueError("Supplied points do not lie on this analytical front")

    m = mop.M
    decision = np.full((len(front), mop.N), 0.5, dtype=np.float64)
    if name == "DTLZ1":
        prefix = np.cumsum(front, axis=1)
        for j in range(m - 1):
            # For DTLZ1: x_j = (f_1+...+f_(M-j-1)) /
            #                  (f_1+...+f_(M-j)).
            numerator = prefix[:, m - j - 2]
            denominator = prefix[:, m - j - 1]
            decision[:, j] = np.divide(
                numerator, denominator,
                out=np.zeros_like(numerator), where=denominator > 0.0
            )
    else:
        angles = _hypersphere_directions_to_angles(np.maximum(front, 0.0))
        scaled = angles / (np.pi / 2.0)
        if name == "DTLZ4":
            # DTLZ4 maps x**100 to angular coordinates.
            scaled = np.power(scaled, 1.0 / 100.0)
        decision[:, :m - 1] = scaled

    if np.any(decision < -1e-12) or np.any(decision > 1 + 1e-12):
        raise ValueError("Computed Pareto-set preimage escaped decision bounds")
    decision = np.clip(decision, 0.0, 1.0)
    mapped = np.asarray(mop.evaluation(decision)["F"], dtype=np.float64)
    if not np.allclose(mapped, front, rtol=0.0, atol=1e-10):
        raise ValueError("Inverse/forward consistency failed for the given MOP")
    return decision


def compare_preimages(name: str, m: int, n: int, *, candidate_pool: int,
                      reference_size: int, seed: int, k: int | None = None) -> dict:
    """One seed, one budget: evaluate genuine MOP preimages and record diagnostics."""
    if name not in _MOPS or m < 2 or n < m or candidate_pool < n or reference_size < 2:
        raise ValueError("Invalid problem, budget or sampling sizes")
    kwargs = {"M": m}
    if k is not None:
        kwargs["K"] = k
    mop = _MOPS[name](**kwargs)

    reference = independent_reference(name, reference_size, m, seed=seed + 1000003)
    start = time.perf_counter()
    old = np.asarray(mop.pf(n_points=n), dtype=np.float64)
    old_seconds = time.perf_counter() - start

    start = time.perf_counter()
    fps = farthest_candidates(name, m, n, candidates=candidate_pool, seed=seed)
    decision = analytical_preimages(mop, fps)
    returned = np.asarray(mop.evaluation(decision)["F"], dtype=np.float64)
    fps_seconds = time.perf_counter() - start

    old_stats = geometry_stats(old, reference, name)
    new_stats = geometry_stats(returned, reference, name)
    return {
        "problem": name, "M": m, "N": mop.N,
        "K": getattr(mop, "K", None), "budget": n,
        "seed": seed, "candidate_pool": candidate_pool,
        "reference_size": reference_size,
        "incumbent": old_stats, "fps_preimage": new_stats,
        "preimage_max_evaluation_error": float(np.max(np.abs(returned - fps))),
        "elapsed_seconds_diagnostic": {
            "incumbent_pf": old_seconds,
            "fps_plus_preimage_plus_evaluation": fps_seconds,
        },
    }


def aggregate(records: list[dict]) -> list[dict]:
    """Summarize repeated measurements without inventing confidence thresholds."""
    groups: dict[tuple[str, int, int], list[dict]] = {}
    for record in records:
        key = (record["problem"], record["M"], record["budget"])
        groups.setdefault(key, []).append(record)

    out = []
    for (name, m, n), group in sorted(groups.items()):
        items = []
        for field in ("reference_to_front_mean", "reference_to_front_max",
                      "nearest_neighbor_cv", "extrema_max_distance"):
            left = np.array([r["incumbent"][field] for r in group], dtype=float)
            right = np.array([r["fps_preimage"][field] for r in group], dtype=float)
            items.append({
                "measure": field,
                "incumbent_median": float(np.median(left)),
                "fps_median": float(np.median(right)),
                "fps_lower_count": int(np.sum(right < left)),
                "pairs": len(group),
            })
        out.append({"problem": name, "M": m, "budget": n, "measures": items,
                    "preimage_max_error": max(r["preimage_max_evaluation_error"]
                                              for r in group)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", choices=list(_MOPS),
                        default=list(_MOPS))
    parser.add_argument("--objectives", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--candidate-pool", type=int, default=4096)
    parser.add_argument("--reference-size", type=int, default=8192)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    records = [
        compare_preimages(name, m, n, candidate_pool=args.candidate_pool,
                          reference_size=args.reference_size, seed=seed, k=args.k)
        for name in args.problems for m in args.objectives
        for n in args.budgets for seed in args.seeds
    ]
    print(json.dumps({
        "schema": "gt_preimage_study_v1", "records": records,
        "summary": aggregate(records),
        "status": "exploratory; not a certified continuous-PF error bound",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
