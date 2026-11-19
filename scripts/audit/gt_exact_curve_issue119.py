# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Continuous Pareto-curve mean distance and exact covering radius for M=2.

For DTLZ1 the PF is a length-sqrt(2)/2 segment between (0.5,0)
and (0,0.5). For DTLZ2/3/4 the PF is the positive unit quarter-circle.
Nearest-point Voronoi cells are intervals in arclength, allowing exact
integration of Euclidean nearest-sample distances, including endpoints.

This is an independent, read-only oracle for the stated curves ONLY.
No general many-objective continuous covering guarantee is implied.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.audit.gt_geometry_pilot import _MOPS, independent_reference, geometry_stats
from scripts.audit.gt_fps_comparison import farthest_candidates
from scripts.audit.gt_voronoi_controls_issue119 import projected_mc_cvt
from scripts.audit.gt_fps_preimage_study import analytical_preimages


def exact_curve_stats(points: np.ndarray, name: str) -> dict:
    """Compute continuous uniform-arclength mean distance and max gap.

    Formula integrates the distance to the nearest sampled objective
    vector in the ambient Euclidean metric, not a parameter-space
    distance. Every discrete point must lie on the analytical curve.
    """
    if name not in ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"):
        raise ValueError("Analytical curve only implemented for DTLZ1-4")
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all():
        raise ValueError("Expected finite nonempty (n,2) objective points")
    if np.any(p < -1e-12):
        raise ValueError("Expected nonnegative Pareto objectives")
    if name == "DTLZ1":
        if not np.allclose(p.sum(axis=1), 0.5, atol=1e-10, rtol=0):
            raise ValueError("Off-simplex GT")
        t = np.sort(np.clip(p[:, 1] / 0.5, 0, 1))
        boundaries = np.concatenate(([0.0], 0.5 * (t[:-1]+t[1:]), [1.0]))
        left = t - boundaries[:-1]
        right = boundaries[1:] - t
        scale = np.sqrt(2.0) / 2.0
        exact_mean = scale * 0.5 * float(np.sum(left*left + right*right))
        edge = max(t[0], 1.0-t[-1])
        interior = float(np.max(np.diff(t))/2.0) if len(t)>1 else 0.0
        exact_radius = scale * max(edge, interior)
    else:
        if not np.allclose(np.linalg.norm(p,axis=1), 1.0, atol=1e-10, rtol=0):
            raise ValueError("Off-hypersphere GT")
        t = np.sort(np.arctan2(np.maximum(p[:,1],0),np.maximum(p[:,0],0)))
        end = np.pi/2
        boundaries = np.concatenate(([0.0], 0.5*(t[:-1]+t[1:]), [end]))
        left = t - boundaries[:-1]
        right = boundaries[1:] - t
        # Integral of 2*sin(|theta-center|/2) dtheta = 4(1-cos(delta/2)).
        exact_mean = float(np.sum(4*(1-np.cos(left/2))+
                                  4*(1-np.cos(right/2))) / end)
        edge = max(2*np.sin(t[0]/2),2*np.sin((end-t[-1])/2))
        interior = 2*np.sin(float(np.max(np.diff(t)))/4) if len(t)>1 else 0.0
        exact_radius = float(max(edge, interior))
    if not np.isfinite(exact_mean) or exact_mean < 0:
        raise ValueError("Invalid continuous mean")
    return {
        "continuous_surface_measure": "uniform_arclength",
        "continuous_mean_euclidean_nearest_distance": exact_mean,
        "exact_continuous_covering_radius": float(exact_radius),
        "endpoints_included": bool(t[0] <= 1e-12 and
                                   abs(t[-1]-(1 if name=="DTLZ1" else np.pi/2)) <= 1e-12),
        "point_count": len(p),
    }


def compare(name: str, n: int, *, candidate_pool: int, seed: int,
            ref_sizes: tuple[int, ...], iterations: int = 12) -> dict:
    """Compare actual MOP and research candidate arrays against M=2 truth."""
    if name not in ("DTLZ1","DTLZ2") or n<2 or candidate_pool<n:
        raise ValueError("Use DTLZ1/2, M=2, and pool>=n>=2")
    mop = _MOPS[name](M=2)
    incumbent = np.asarray(mop.pf(n_points=n), dtype=float)
    fps = farthest_candidates(name,2,n,candidates=candidate_pool,seed=seed)
    pool = independent_reference(name,candidate_pool,2,seed)
    hybrid = projected_mc_cvt(name,fps,pool,iterations=iterations)
    samples={"incumbent":incumbent,"fps":fps,"fps_mc_cvt":hybrid}
    records={}
    for method,points in samples.items():
        if method!="incumbent":
            x = analytical_preimages(mop,points)
            actual=np.asarray(mop.evaluation(x)["F"],dtype=float)
            if not np.allclose(actual,points,rtol=0,atol=1e-10):
                raise ValueError("Invalid PS inverse")
            points=actual
        continuous = exact_curve_stats(points,name)
        estimates={}
        for size in ref_sizes:
            ref=independent_reference(name,size,2,seed+1_000_003)
            stats=geometry_stats(points,ref,name)
            estimates[str(size)]={
                "sampled_mean":stats["reference_to_front_mean"],
                "sampled_max_lower_bound":stats["reference_to_front_max"],
                "mean_absolute_error":abs(stats["reference_to_front_mean"]-
                                         continuous["continuous_mean_euclidean_nearest_distance"]),
                "max_gap_to_exact":continuous["exact_continuous_covering_radius"]-
                                   stats["reference_to_front_max"],
            }
        records[method]={"exact":continuous,"reference_estimates":estimates}
    return {"name":name,"M":2,"n":n,"candidate_pool":candidate_pool,
            "seed":seed,"methods":records}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budgets",type=int,nargs="+",default=[64,256,1024])
    p.add_argument("--seeds",type=int,nargs="+",default=[101,202,303])
    p.add_argument("--pool",type=int,default=4096)
    p.add_argument("--ref-sizes",type=int,nargs="+",default=[2048,8192])
    p.add_argument("--iterations",type=int,default=12)
    a=p.parse_args()
    data=[compare(name,n,candidate_pool=a.pool,seed=seed,
                  ref_sizes=tuple(a.ref_sizes),iterations=a.iterations)
          for name in ("DTLZ1","DTLZ2") for n in a.budgets for seed in a.seeds]
    print(json.dumps({"schema":"gt119_exact_m2_curve_v1",
                     "scope":"proved continuous segment/quarter-circle only",
                     "records":data},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
