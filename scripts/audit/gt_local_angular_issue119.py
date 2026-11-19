# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #119: independent, read-only *local* angular covering certificate.

The positive hypersphere's angular map has orthogonal Jacobian columns,
with derivative norm for angle j equal to product(cos(theta_i),i<j).
Inside [lo,hi], each norm is <= product(cos(lo_i),i<j). Hence
  ||f(theta)-f(mid)|| <= ||weights(lo)*(hi-lo)/2||_2.
This sharpens the *local* cell bound without increasing its point count.
The active cell with largest upper bound is bisected along the angular
coordinate having largest weighted half-width. Parent caps are inherited.

Mathematical claim is for ideal real arithmetic, NOT an interval-verified
floating-point enclosure. This does not modify the existing certificate.
"""
from __future__ import annotations

import argparse
import heapq
import json
from math import pi, sqrt
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from scripts.audit.gt_angular_cover_issue119 import (
    angular_coverage_interval, positive_sphere_angles,
)
from scripts.audit.gt_geometry_pilot import _MOPS
from scripts.audit.gt_fps_comparison import farthest_candidates
from scripts.audit.gt_fps_preimage_study import analytical_preimages
from scripts.audit.gt_refined_net_issue119 import improved_coverage_interval


def local_weights(lo: np.ndarray) -> np.ndarray:
    """Upper norm bounds on each angular Jacobian column on this cell."""
    lo = np.asarray(lo, dtype=float)
    if lo.ndim != 1 or len(lo) < 1 or not np.isfinite(lo).all():
        raise ValueError("Expected finite angular lower bounds")
    if np.any(lo < 0) or np.any(lo > pi / 2):
        raise ValueError("Angles out of positive orthant")
    return np.r_[1.0, np.cumprod(np.cos(lo[:-1]))]


def local_radius(lo: np.ndarray, hi: np.ndarray) -> float:
    """Guaranteed Euclidean chord radius about the angular cell center."""
    a = np.asarray(lo, dtype=float)
    b = np.asarray(hi, dtype=float)
    if (a.shape != b.shape or a.ndim != 1 or len(a) < 1 or
            not np.isfinite(b).all() or np.any(b < a) or np.any(b > pi / 2)):
        raise ValueError("Invalid angular cell")
    return float(np.linalg.norm(local_weights(a) * (b-a) / 2))


def tighter_angular_interval(front: np.ndarray, *, max_cells: int = 4000,
                             width_target: float = 0.01) -> dict:
    """Continuous worst-distance interval from locally weighted branch/bound."""
    gt=np.asarray(front,dtype=float)
    if (gt.ndim!=2 or len(gt)<1 or gt.shape[1]<2 or
            not np.isfinite(gt).all() or np.any(gt < -1e-12) or
            not np.allclose(np.linalg.norm(gt,axis=1),1,atol=1e-10,rtol=0)):
        raise ValueError("Expected finite nonempty positive unit-sphere GT")
    if max_cells < 1 or not np.isfinite(width_target) or width_target <= 0:
        raise ValueError("Need max_cells>=1 and width_target>0")
    tree=cKDTree(gt)
    m=gt.shape[1]
    count=0

    def bound(lo,hi,cap):
        nonlocal count
        center=positive_sphere_angles((lo+hi)/2)
        d=float(tree.query(center,k=1,workers=1)[0])
        count+=1
        return d,min(cap,d+local_radius(lo,hi))

    a=np.zeros(m-1)
    b=np.full(m-1,pi/2)
    d,u=bound(a,b,sqrt(2))
    lower=d
    queue=[(-u,0,a,b)]
    serial=0
    start=perf_counter()
    while len(queue)<max_cells and -queue[0][0]-lower>width_target:
        neg,_,lo,hi=heapq.heappop(queue)
        parent_upper=-neg
        # Split along greatest possible local Euclidean contribution.
        axis=int(np.argmax(local_weights(lo)*(hi-lo)))
        midpoint=(lo[axis]+hi[axis])/2
        left_hi=hi.copy();left_hi[axis]=midpoint
        right_lo=lo.copy();right_lo[axis]=midpoint
        for x,y in ((lo,left_hi),(right_lo,hi)):
            d,u=bound(x,y,parent_upper)
            lower=max(lower,d)
            serial+=1
            heapq.heappush(queue,(-u,serial,x,y))
    upper=max(lower,-queue[0][0])
    return {
        "M":m,"GT_points":len(gt),
        "continuous_radius_lower":float(lower),
        "continuous_radius_upper":float(upper+1e-12),
        "interval_width":float(upper-lower+1e-12),
        "angular_cells":len(queue),"evaluated_centers":count,
        "cell_budget":max_cells,"width_target":width_target,
        "elapsed_seconds_diagnostic":perf_counter()-start,
        "mathematical_guarantee":"local orthogonal Jacobian derivative bound in real arithmetic",
        "floating_point_enclosure_verified":False,
    }


def compare(m: int,n: int,*,q: int,cell_budget: int,seed: int,
            candidate_pool: int,target: float=0.01) -> dict:
    if m<3 or n<m or candidate_pool<n:
        raise ValueError("Need M>=3, candidate_pool>=n>=M")
    mop=_MOPS["DTLZ2"](M=m)
    actual=np.asarray(mop.pf(n_points=n),dtype=float)
    fps=farthest_candidates("DTLZ2",m,n,candidates=candidate_pool,seed=seed)
    x=analytical_preimages(mop,fps)
    valid=np.asarray(mop.evaluation(x)["F"],dtype=float)
    if not np.allclose(valid,fps,atol=1e-10,rtol=0):
        raise ValueError("Failed MOP preimage replay")
    methods={}
    for method,gt in (("incumbent",actual),("fps",valid)):
        old=angular_coverage_interval(gt,max_cells=cell_budget,
                                      width_target=target)
        new=tighter_angular_interval(gt,max_cells=cell_budget,
                                     width_target=target)
        bary=improved_coverage_interval(gt,"DTLZ2",q=q)
        lower=max(new["continuous_radius_lower"],
                  old["continuous_radius_lower"],
                  bary["observed_net_max_lower"])
        upper=min(new["continuous_radius_upper"],
                  old["continuous_radius_upper"],
                  bary["continuous_radius_upper"])
        if lower>upper+1e-10:
            raise ValueError("Independent mathematically valid intervals disagree")
        methods[method]={
            "original_angular":old,
            "local_angular":new,
            "barycentric":{
                "lower":bary["observed_net_max_lower"],
                "upper":bary["continuous_radius_upper"],
                "net_points":bary["net_points"],
            },
            "combined":{"lower":float(lower),"upper":float(max(lower,upper)),
                        "width":float(max(0,upper-lower))},
        }
    return {"problem":"DTLZ2","M":m,"GT_budget":n,
            "seed":seed,"candidate_pool":candidate_pool,
            "barycentric_q":q,"methods":methods}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objectives",type=int,nargs="+",default=[3,5])
    p.add_argument("--budgets",type=int,nargs="+",default=[128,512])
    p.add_argument("--seeds",type=int,nargs="+",default=[101,202])
    p.add_argument("--q",type=int,default=24)
    p.add_argument("--cells",type=int,default=4000)
    p.add_argument("--target",type=float,default=0.01)
    p.add_argument("--candidate-pool",type=int,default=4096)
    a=p.parse_args()
    records=[compare(m,n,q=a.q,cell_budget=a.cells,seed=seed,
                     candidate_pool=a.candidate_pool,target=a.target)
             for m in a.objectives for n in a.budgets for seed in a.seeds]
    print(json.dumps({"schema":"gt119_local_angular_v1",
                      "status":"read-only real-arithmetic bounds; FP not interval-certified",
                      "records":records},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
