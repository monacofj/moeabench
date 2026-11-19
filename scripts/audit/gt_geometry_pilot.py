# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only, independent objective-space pilot for analytical GT samplers (#119).

No MOP calibration, stored GT, random global state, regression fixture or
numerical-performance baseline is read or changed. Run from repository root:
    python scripts/audit/gt_geometry_pilot.py --budget 256 --budget 1024
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial import cKDTree

from moeabench.mops import DTLZ1, DTLZ2, DTLZ3, DTLZ4


_MOPS = {"DTLZ1": DTLZ1, "DTLZ2": DTLZ2, "DTLZ3": DTLZ3, "DTLZ4": DTLZ4}


def independent_reference(name: str, n: int, m: int, seed: int) -> np.ndarray:
    """Generate independently defined, surface-uniform objective-space points.

    For the DTLZ1 simplex, normalized independent exponentials give Dirichlet(1).
    For the positive hypersphere, normalized absolute Gaussians give uniform
    surface directions. These laws follow directly from geometry, not from
    MOP.ps() or the implementation of MOP.evaluation().
    """
    if n < 1 or m < 2 or name not in _MOPS:
        raise ValueError("Expected supported DTLZ name, n>=1, M>=2")
    rng = np.random.default_rng(seed)
    if name == "DTLZ1":
        e = rng.exponential(size=(n, m))
        return 0.5 * e / e.sum(axis=1, keepdims=True)
    g = np.abs(rng.standard_normal(size=(n, m)))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def _positive_nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([], dtype=float)
    distances, _ = cKDTree(points).query(points, k=2)
    return np.asarray(distances[:, 1], dtype=float)


def geometry_stats(front: np.ndarray, reference: np.ndarray, name: str) -> dict:
    """Describe actual coverage and spacing; no heuristic PASS threshold."""
    front = np.asarray(front, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if front.ndim != 2 or reference.ndim != 2 or front.shape[1] != reference.shape[1]:
        raise ValueError("Expected matching 2D objective-space point matrices")
    if len(front) < 2 or len(reference) < 2:
        raise ValueError("At least two points required in each set")
    if not np.isfinite(front).all() or not np.isfinite(reference).all():
        raise ValueError("All points must be finite")
    m = front.shape[1]
    target = 0.5 if name == "DTLZ1" else 1.0
    geometry = front.sum(axis=1) if name == "DTLZ1" else np.linalg.norm(front, axis=1)
    reference_to_front = cKDTree(front).query(reference, k=1)[0]
    front_to_reference = cKDTree(reference).query(front, k=1)[0]
    nearest = _positive_nearest_neighbor_distances(front)
    mean_nn = float(np.mean(nearest))
    extrema = target * np.eye(m)
    extent_error = cKDTree(front).query(extrema, k=1)[0]
    return {
        "n": int(len(front)),
        "M": int(m),
        "unique_rows": int(len(np.unique(front, axis=0))),
        "geometry_max_residual": float(np.max(np.abs(geometry - target))),
        "reference_to_front_mean": float(np.mean(reference_to_front)),
        "reference_to_front_p95": float(np.percentile(reference_to_front, 95)),
        "reference_to_front_max": float(np.max(reference_to_front)),
        "front_to_reference_mean": float(np.mean(front_to_reference)),
        "nearest_neighbor_mean": mean_nn,
        "nearest_neighbor_cv": float(np.std(nearest) / mean_nn) if mean_nn > 0 else None,
        "extrema_max_distance": float(np.max(extent_error)),
    }


def audit(name: str, m: int, budget: int, *, reference_size: int, seed: int) -> dict:
    if budget < 2 or reference_size < 2:
        raise ValueError("Need budget>=2 and reference_size>=2")
    mop = _MOPS[name](M=m)
    front = np.asarray(mop.pf(n_points=budget), dtype=float)
    # Reference seed is distinct from the implementation's historical seed=42.
    reference = independent_reference(name, reference_size, m, seed=seed)
    result = geometry_stats(front, reference, name)
    return {"problem": name, "budget": budget, "reference_size": reference_size,
            "reference_seed": seed, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", action="append", type=int, dest="budgets",
                        help="Candidate front budget; repeat for multiple budgets")
    parser.add_argument("--reference-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--objectives", type=int, nargs="+", default=[3, 5])
    args = parser.parse_args()
    budgets = args.budgets or [256, 1024]
    records = [
        audit(name, m, budget, reference_size=args.reference_size, seed=args.seed)
        for name in _MOPS for m in args.objectives for budget in budgets
    ]
    print(json.dumps({"schema": "gt_geometry_pilot_v1",
                      "purpose": "diagnostic; no pass/fail or reference replacement",
                      "records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
