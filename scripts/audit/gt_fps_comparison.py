# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only controlled FPS candidate comparison for issue #119.

Run:
 python scripts/audit/gt_fps_comparison.py --objectives 3 5 --budget 256 1024

The experiment compares unchanged MOP.pf() with a deliberately separate
objective-space farthest-point selection; it does not replace PS/PF, calibrate,
write baselines, modify random global state, or certify covering radius over
the continuous front. Reference-to-set distances are finite-sample estimates.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from scripts.audit.gt_geometry_pilot import (
    _MOPS,
    geometry_stats,
    independent_reference,
)


def farthest_candidates(name: str, m: int, n: int, *, candidates: int,
                        seed: int) -> np.ndarray:
    """Greedy Euclidean farthest-point selection from independent candidates.

    Preserve all objective-axis extrema; candidate density is an experimental
    parameter and is NOT a certificate for covering the continuous geometry.
    """
    if name not in _MOPS or m < 2 or n < m or candidates < n:
        raise ValueError("Expected supported MOP, M>=2 and candidates>=n>=M")
    points = independent_reference(name, candidates, m, seed=seed)
    target = 0.5 if name == "DTLZ1" else 1.0
    extrema = target * np.eye(m)
    output = np.empty((n, m), dtype=float)
    output[:m] = extrema
    minimum_squared = np.full(candidates, np.inf)
    for axis in extrema:
        minimum_squared = np.minimum(
            minimum_squared,
            np.einsum("ij,ij->i", points - axis, points - axis),
        )
    for j in range(m, n):
        idx = int(np.argmax(minimum_squared))
        selected = points[idx]
        output[j] = selected
        minimum_squared = np.minimum(
            minimum_squared,
            np.einsum("ij,ij->i", points - selected, points - selected),
        )
        minimum_squared[idx] = -1.0
    return output


def compare(name: str, m: int, n: int, *, candidates: int,
            reference_size: int, seed: int) -> dict:
    """Run a controlled comparison; independent reference never used to choose P."""
    if n < m or reference_size < 2 or candidates < n:
        raise ValueError("Need n>=M, reference_size>=2, candidates>=n")
    reference = independent_reference(name, reference_size, m, seed=seed + 1)
    start = time.perf_counter()
    incumbent = np.asarray(_MOPS[name](M=m).pf(n_points=n), dtype=float)
    incumbent_seconds = time.perf_counter() - start
    start = time.perf_counter()
    candidate = farthest_candidates(name, m, n, candidates=candidates, seed=seed)
    candidate_seconds = time.perf_counter() - start
    return {
        "problem": name, "M": m, "budget": n, "reference_size": reference_size,
        "candidate_pool": candidates, "candidate_seed": seed,
        "reference_seed": seed + 1,
        "incumbent": geometry_stats(incumbent, reference, name),
        "candidate_fps": geometry_stats(candidate, reference, name),
        "elapsed_seconds_noncertifying": {
            "incumbent": incumbent_seconds, "candidate_fps": candidate_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--budget", nargs="+", type=int, default=[256, 1024])
    parser.add_argument("--candidate-pool", type=int, default=8192)
    parser.add_argument("--reference-size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--problems", nargs="+", choices=list(_MOPS),
                        default=list(_MOPS))
    args = parser.parse_args()
    records = [
        compare(name, m, n, candidates=args.candidate_pool,
                reference_size=args.reference_size, seed=args.seed)
        for name in args.problems for m in args.objectives for n in args.budget
    ]
    print(json.dumps({
        "schema": "gt_fps_comparison_v1",
        "status": "exploratory, read-only; no continuous covering certificate",
        "records": records,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
