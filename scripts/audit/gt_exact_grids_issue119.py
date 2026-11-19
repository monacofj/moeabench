# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Research-only exact, deterministic GT grids for DTLZ1/2 with M=2.

Compare current MOP.pf, FPS, and two closed-form arclength grids:
- endpoint-constrained equal-spacing, optimal for max radius and mean
  nearest Euclidean distance among n-point grids containing both axes;
- unconstrained mid-cell positions, optimal for max radius and mean
  nearest Euclidean distance among arbitrary n-point curve samples.

The claims are ONLY for the continuous segment and positive quarter-circle
under uniform arclength, with n>=2. No code or regression is changed.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from scripts.audit.gt_geometry_pilot import _MOPS
from scripts.audit.gt_exact_curve_issue119 import exact_curve_stats
from scripts.audit.gt_fps_comparison import farthest_candidates
from scripts.audit.gt_fps_preimage_study import analytical_preimages


def exact_arclength_grid(name: str, n: int, *, endpoints: bool) -> np.ndarray:
    if name not in ("DTLZ1", "DTLZ2") or n < 2:
        raise ValueError("Only DTLZ1/2 M=2 and n>=2")
    t = np.linspace(0.0, 1.0, n) if endpoints else (np.arange(n)+0.5)/n
    if name == "DTLZ1":
        return np.column_stack((0.5 * (1-t), 0.5 * t))
    theta=t*np.pi/2
    return np.column_stack((np.cos(theta), np.sin(theta)))


def theoretical_optimum(name: str, n: int, *, endpoints: bool) -> dict:
    if name not in ("DTLZ1", "DTLZ2") or n<2:
        raise ValueError("Only 1D DTLZ1/2; n>=2")
    cells = n-1 if endpoints else n
    if name=="DTLZ1":
        length=np.sqrt(2.0)/2.0
        mean=length/(4*cells)
        radius=length/(2*cells)
    else:
        length=np.pi/2
        gap=length/cells
        # one gap with endpoint samples contributes 8[1-cos(gap/4)]
        # to the arclength distance integral, or 2x the corresponding
        # midpoint-cell integral; both designs lead to this same formula
        # averaged over the full arc.
        mean=8*(1-np.cos(gap/4))/gap
        radius=2*np.sin(gap/4)
    return {"continuous_mean_euclidean_nearest_distance":float(mean),
            "exact_continuous_covering_radius":float(radius)}


def compare(name: str, n: int, *, seed: int, pool_size: int) -> dict:
    if name not in ("DTLZ1", "DTLZ2") or n<2 or pool_size<n:
        raise ValueError("Invalid name/point budget")
    mop=_MOPS[name](M=2)
    results={}
    configs={
        "incumbent":lambda:np.asarray(mop.pf(n_points=n),dtype=float),
        "fps":lambda:farthest_candidates(name,2,n,candidates=pool_size,seed=seed),
        "equal_arclength_with_axes":lambda:exact_arclength_grid(name,n,endpoints=True),
        "equal_arclength_midpoints":lambda:exact_arclength_grid(name,n,endpoints=False),
    }
    for method,fn in configs.items():
        start=time.perf_counter()
        points=fn()
        if method!="incumbent":
            ps=analytical_preimages(mop,points)
            checked=np.asarray(mop.evaluation(ps)["F"],dtype=float)
            if not np.allclose(checked,points,atol=1e-10,rtol=0):
                raise ValueError("Analytical preimage validation failed")
            points=checked
        cost=time.perf_counter()-start
        observed=exact_curve_stats(points,name)
        row={"observed":observed,"complete_generation_seconds":cost}
        if method.startswith("equal_arclength"):
            expected=theoretical_optimum(name,n,endpoints=method.endswith("with_axes"))
            row["theoretical_optimum"]=expected
            if any(abs(expected[k]-observed[k])>1e-10 for k in expected):
                raise ValueError("Grid does not match exact analytic optimum")
        results[method]=row
    return {"problem":name,"M":2,"budget":n,"candidate_pool":pool_size,
            "seed":seed,"methods":results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budgets",nargs="+",type=int,default=[64,256,1024])
    p.add_argument("--seeds",nargs="+",type=int,default=[101,202,303])
    p.add_argument("--pool",type=int,default=4096)
    a=p.parse_args()
    data=[compare(name,n,seed=seed,pool_size=a.pool)
          for name in ("DTLZ1","DTLZ2") for n in a.budgets for seed in a.seeds]
    print(json.dumps({"schema":"gt119_exact_m2_grids_v1",
                      "status":"research, no production change",
                      "records":data},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
