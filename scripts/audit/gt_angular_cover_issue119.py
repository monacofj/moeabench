# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only adaptive angular covering certificate for positive-sphere PFs.

For DTLZ2/3/4 the positive unit sphere is parameterized by M-1 angles
theta in [0,pi/2]. The coordinate tangent vectors are mutually
orthogonal and each has norm <= 1, so the map is 1-Lipschitz from
Euclidean angle space to objective-space chord distance.

For an angular rectangular cell C, center c and half-diagonal r,
  max_{theta in C} dist(f(theta), GT) <= dist(f(c), GT) + r.
A center is on the continuous front, and its distance gives a lower
bound. Recursively split the region with the largest upper bound;
inherit the parent upper bound (children are subsets) to tighten it.

The resulting lower/upper enclosure is valid in exact real arithmetic.
Floating-point KD-tree queries and the 1e-12 allowance are not a
formally verified interval-arithmetic implementation.

No MOP, calibration sidecars, historical regressions or GTs are changed.
"""
from __future__ import annotations

import argparse
import heapq
import json
from math import pi, sqrt
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from scripts.audit.gt_geometry_pilot import _MOPS
from scripts.audit.gt_fps_comparison import farthest_candidates
from scripts.audit.gt_fps_preimage_study import analytical_preimages
from scripts.audit.gt_refined_net_issue119 import improved_coverage_interval


def positive_sphere_angles(theta: np.ndarray) -> np.ndarray:
    """Exact hyperspherical positive-orthant mapping for M=len(theta)+1."""
    t = np.asarray(theta, dtype=float)
    if t.ndim != 1 or len(t) < 1 or not np.isfinite(t).all():
        raise ValueError("Angles must be a finite (M-1,) array")
    if np.any(t < 0) or np.any(t > pi / 2):
        raise ValueError("Angles must lie in [0, pi/2]")
    m = len(t) + 1
    p = np.empty(m, dtype=float)
    c = np.cos(t)
    p[0] = np.prod(c)
    for i in range(1, m):
        j = m - 1 - i
        p[i] = np.prod(c[:j]) * np.sin(t[j])
    return p


def angular_coverage_interval(front: np.ndarray, *,
                              max_cells: int = 4000,
                              width_target: float = 0.01) -> dict:
    """Adaptive branch-and-bound bound on continuous worst GT distance.

    max_cells limits number of ACTIVE hyperrectangles, not evaluated
    centers; the actual count is reported. Repeated refinements use a
    priority queue and do not create a full factorial angle lattice.
    """
    f = np.asarray(front, dtype=float)
    if (f.ndim != 2 or len(f) < 1 or f.shape[1] < 2 or
            not np.isfinite(f).all() or np.any(f < -1e-12) or
            not np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError("Require nonempty valid positive unit-sphere GT")
    if max_cells < 1 or not np.isfinite(width_target) or width_target <= 0:
        raise ValueError("Require max_cells>=1 and width_target>0")
    m = f.shape[1]
    tree = cKDTree(f)
    query_count = 0

    def node(lo: np.ndarray, hi: np.ndarray, cap: float):
        nonlocal query_count
        mid = (lo + hi) / 2.0
        geom = positive_sphere_angles(mid)
        distance = float(tree.query(geom, k=1, workers=1)[0])
        query_count += 1
        bound = distance + float(np.linalg.norm(hi - lo)) / 2.0
        return distance, min(cap, bound)

    lo = np.zeros(m - 1)
    hi = np.full(m - 1, pi/2)
    first_dist, first_upper = node(lo, hi, sqrt(2))
    lower = first_dist
    serial = 0
    # (-upper, serial, lo, hi); serial breaks ties without comparing arrays.
    queue = [(-first_upper, serial, lo, hi)]
    start = perf_counter()
    while len(queue) < max_cells and -queue[0][0] - lower > width_target:
        neg_upper, _, a, b = heapq.heappop(queue)
        parent_upper = -neg_upper
        axis = int(np.argmax(b - a))
        center = (a[axis]+b[axis])/2.0
        left_high=b.copy()
        left_high[axis]=center
        right_low=a.copy()
        right_low[axis]=center
        for child_low,child_high in ((a,left_high),(right_low,b)):
            dist,upper=node(child_low,child_high,parent_upper)
            lower=max(lower,dist)
            serial+=1
            heapq.heappush(queue,(-upper,serial,child_low,child_high))
    upper=max(lower,-queue[0][0])
    return {
        "M":m,"GT_points":len(f),
        "continuous_radius_lower":float(lower),
        "continuous_radius_upper":float(upper+1e-12),
        "interval_width":float(upper-lower+1e-12),
        "angular_cells":len(queue),
        "evaluated_centers":query_count,
        "width_target":float(width_target),
        "cell_budget":int(max_cells),
        "elapsed_seconds_diagnostic":perf_counter()-start,
        "mathematical_guarantee":"1-Lipschitz angular map, real arithmetic",
        "floating_point_enclosure_verified":False,
    }


def compare(m: int, n: int, *, q: int, cell_budget: int, seed: int,
            candidate_pool: int, target: float = 0.01) -> dict:
    if m < 3 or n < m or candidate_pool < n:
        raise ValueError("Require M>=3 and candidate pool>=n>=M")
    mop = _MOPS["DTLZ2"](M=m)
    baseline = np.asarray(mop.pf(n_points=n),dtype=float)
    fps = farthest_candidates("DTLZ2",m,n,candidates=candidate_pool,seed=seed)
    decision = analytical_preimages(mop,fps)
    replay = np.asarray(mop.evaluation(decision)["F"],dtype=float)
    if not np.allclose(fps,replay,atol=1e-10,rtol=0):
        raise ValueError("Objective/preimage validation failed")
    results = {}
    for label,gt in (("incumbent",baseline),("fps",replay)):
        bary=improved_coverage_interval(gt,"DTLZ2",q=q)
        angular=angular_coverage_interval(gt,max_cells=cell_budget,
                                          width_target=target)
        # Intersection of independently valid real-arithmetic intervals is
        # itself valid and can only improve their tightness. This is especially
        # useful when angular refinement stalls at higher intrinsic dimension.
        intersection_lower = max(bary["observed_net_max_lower"],
                                 angular["continuous_radius_lower"])
        intersection_upper = min(bary["continuous_radius_upper"],
                                 angular["continuous_radius_upper"])
        if intersection_lower > intersection_upper + 1e-10:
            raise ValueError("Independent bounds disagree: check numerical geometry")
        results[label]={
            "combined":{
                "lower":float(intersection_lower),
                "upper":float(max(intersection_lower, intersection_upper)),
                "width":float(max(0.0,intersection_upper-intersection_lower)),
                "construction":"intersection of independent valid bounds",
            },
            "barycentric":{
                "lower":bary["observed_net_max_lower"],
                "upper":bary["continuous_radius_upper"],
                "width":bary["interval_width"],
                "net_points":bary["net_points"],
            },
            "angular":angular,
        }
    return {"problem":"DTLZ2","M":m,"budget":n,"seed":seed,
            "candidate_pool":candidate_pool,"barycentric_q":q,
            "methods":results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives",type=int,nargs="+",default=[3,5])
    parser.add_argument("--budgets",type=int,nargs="+",default=[128,512])
    parser.add_argument("--seeds",type=int,nargs="+",default=[101,202])
    parser.add_argument("--q",type=int,default=24)
    parser.add_argument("--cells",type=int,default=4000)
    parser.add_argument("--target",type=float,default=0.01)
    parser.add_argument("--candidate-pool",type=int,default=4096)
    a=parser.parse_args()
    records=[compare(m,n,q=a.q,cell_budget=a.cells,seed=seed,
                     candidate_pool=a.candidate_pool,target=a.target)
             for m in a.objectives for n in a.budgets for seed in a.seeds]
    print(json.dumps({
        "schema":"gt119_adaptive_angular_v1",
        "status":"research-only real-arithmetic guaranteed coverage bounds",
        "records":records,
    },sort_keys=True,indent=2))


if __name__=="__main__":
    main()
