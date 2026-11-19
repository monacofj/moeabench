# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only certified continuous covering intervals for DTLZ1-4, M>=3.

A finite barycentric lattice is an epsilon-net of the simplex. Its
normalization is an epsilon-net of the positive unit sphere. For an
arbitrary fixed valid GT set S, nearest-set distance is 1-Lipschitz:
  max_{v in net} d(v,S) <= sup_{x in PF} d(x,S)
                       <= max_{v in net} d(v,S) + epsilon.
Unlike random finite references, the right side is a proved continuous
upper bound (modulo floating-point roundoff). It may be very loose.

No production sampler, GT artifact, certified baseline or test is touched.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from math import comb
import json

import numpy as np
from scipy.spatial import cKDTree

from scripts.audit.gt_geometry_pilot import _MOPS
from scripts.audit.gt_fps_comparison import farthest_candidates
from scripts.audit.gt_fps_preimage_study import analytical_preimages


def barycentric_lattice(m: int, q: int, *, max_points: int = 100_000) -> np.ndarray:
    """Return every q-level unit-simplex integer composition divided by q."""
    if m < 2 or q < 1:
        raise ValueError("Require M>=2, resolution q>=1")
    count = comb(q + m - 1, m - 1)
    if count > max_points:
        raise ValueError(f"Refusing lattice of {count} points (limit {max_points})")
    result = np.empty((count, m), dtype=float)
    for i, bars in enumerate(combinations(range(q + m - 1), m - 1)):
        separators = (-1, *bars, q + m - 1)
        result[i] = np.diff(separators) - 1
    result /= q
    return result


def analytical_epsilon(name: str, m: int, q: int) -> float:
    """Conservative Euclidean epsilon-net radius from balanced integer rounding.

    For any unit-simplex u, take floors(q*u) and assign leftover units
    to largest fractional parts. Each component error is at most 1/q,
    hence ||u - v||_2 <= sqrt(M)/q. DTLZ1 scales coordinates by 1/2.

    For positive-sphere f=u/||u||, normalization obeys
    ||u/||u|| - v/||v|||| <= ||u-v||/sqrt(||u|| ||v||)
    <= sqrt(M) ||u-v|| because positive simplex vectors have norm
    >= 1/sqrt(M). Therefore epsilon <= M/q for DTLZ2/3/4.
    These bounds remain conservative, especially in higher dimensions.
    """
    if name not in _MOPS or m < 2 or q < 1:
        raise ValueError("Require DTLZ1-4, M>=2, q>=1")
    return (0.5 * np.sqrt(m) if name == "DTLZ1" else float(m)) / q


def coverage_interval(front: np.ndarray, name: str, *, q: int,
                      max_points: int = 100_000) -> dict:
    """Certified two-sided interval on worst CONTINUOUS surface coverage."""
    front = np.asarray(front, dtype=float)
    if name not in _MOPS or front.ndim != 2 or len(front) < 1:
        raise ValueError("Expected nonempty (n,M) DTLZ1-4 GT")
    m = front.shape[1]
    if m < 2 or not np.isfinite(front).all() or np.any(front < -1e-12):
        raise ValueError("GT must have M>=2 finite nonnegative objectives")
    if name == "DTLZ1":
        residual = front.sum(axis=1)
        if not np.allclose(residual, 0.5, atol=1e-10, rtol=0):
            raise ValueError("GT is off analytical simplex")
    else:
        residual = np.linalg.norm(front, axis=1)
        if not np.allclose(residual, 1.0, atol=1e-10, rtol=0):
            raise ValueError("GT is off positive unit sphere")
    grid = barycentric_lattice(m, q, max_points=max_points)
    if name == "DTLZ1":
        net = 0.5 * grid
    else:
        net = grid / np.linalg.norm(grid, axis=1, keepdims=True)
    observed = float(np.max(cKDTree(front).query(net, k=1, workers=1)[0]))
    epsilon = analytical_epsilon(name, m, q)
    # 1e-12 covers arithmetic roundoff in finite computations; not a
    # formally verified floating-point enclosure or interval arithmetic.
    return {
        "M":m, "GT_points":len(front), "q":q, "net_points":len(net),
        "observed_net_max_lower":observed,
        "analytic_net_epsilon":float(epsilon),
        "continuous_radius_upper":float(observed + epsilon + 1e-12),
        "interval_width":float(epsilon + 1e-12),
        "guarantee":"mathematical real-arithmetic bound; FP tolerance 1e-12",
        "mean_distance_certified":False,
    }


def audit(name: str, m: int, n: int, *, q_values: tuple[int,...],
          candidate_pool: int, seed: int, max_points: int) -> dict:
    if name not in ("DTLZ1", "DTLZ2") or m < 3 or n < m or candidate_pool < n:
        raise ValueError("Use DTLZ1/2, M>=3 and candidate_pool>=n")
    mop = _MOPS[name](M=m)
    actual = np.asarray(mop.pf(n_points=n), dtype=float)
    fps = farthest_candidates(name, m, n, candidates=candidate_pool, seed=seed)
    x = analytical_preimages(mop, fps)
    replay = np.asarray(mop.evaluation(x)["F"], dtype=float)
    if not np.allclose(replay, fps, atol=1e-10, rtol=0):
        raise ValueError("FPS inverse did not match original MOP evaluation")
    return {
        "problem":name, "M":m, "GT_budget":n,
        "candidate_pool":candidate_pool, "candidate_seed":seed,
        "methods":{
            "incumbent":{str(q):coverage_interval(actual,name,q=q,max_points=max_points)
                         for q in q_values},
            "fps":{str(q):coverage_interval(replay,name,q=q,max_points=max_points)
                   for q in q_values},
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objectives",type=int,nargs="+",default=[3,5])
    p.add_argument("--budgets",type=int,nargs="+",default=[128,512])
    p.add_argument("--resolutions",type=int,nargs="+",default=[8,16,24])
    p.add_argument("--pool",type=int,default=4096)
    p.add_argument("--seed",type=int,default=101)
    p.add_argument("--max-points",type=int,default=100_000)
    args=p.parse_args()
    data=[audit(name,m,n,q_values=tuple(args.resolutions),
                candidate_pool=args.pool,seed=args.seed,max_points=args.max_points)
          for name in ("DTLZ1","DTLZ2")
          for m in args.objectives for n in args.budgets]
    print(json.dumps({
        "schema":"gt119_analytic_net_covering_v1",
        "scope":"DTLZ1 simplex; DTLZ2/3/4 positive hypersphere",
        "status":"read-only research, conservative analytic upper bounds",
        "records":data,
    },sort_keys=True,indent=2))


if __name__ == "__main__":
    main()
