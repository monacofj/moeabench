# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent sharper real-arithmetic epsilon-net bounds, issue #119.

The previous barycentric-lattice certificate used the componentwise
|rounding error| <= 1/q bound, giving sqrt(M)/q on the simplex. A
dependent-rounding proof gives sqrt(M)/(2q), without changing the net:

For u on the unit simplex let q*u = floor(q*u) + r, with r_i in [0,1)
and integer sum(r)=k. Randomized systematic rounding yields a k-hot
0/1 vector B with E B_i=r_i and sum B_i=k (choose one uniform offset
for the cumulative fractional sums). Thus

  E ||B-r||^2 = sum r_i(1-r_i) <= M/4.

At least one k-hot B has ||B-r|| <= sqrt(M)/2. The complete
q-barycentric lattice contains this rounded composition. Scaling
the simplex by 1/2 gives epsilon <= sqrt(M)/(4q).
The positive sphere uses normalized simplex vectors, and
||u/||u||-v/||v|||| <= ||u-v|| / sqrt(||u|| ||v||)
<= sqrt(M)*||u-v||, giving epsilon <= M/(2q).

This is a rigorous mathematical bound in exact real arithmetic.
The floating-point evaluations are NOT interval-certified and add
only a conservative, non-verified implementation rounding allowance.
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from scripts.audit.gt_certified_net_issue119 import (
    analytical_epsilon as legacy_epsilon,
    coverage_interval as legacy_interval,
)
from scripts.audit.gt_geometry_pilot import _MOPS
from scripts.audit.gt_fps_comparison import farthest_candidates
from scripts.audit.gt_fps_preimage_study import analytical_preimages


def dependent_rounding_epsilon(name: str, m: int, q: int) -> float:
    """Proved epsilon-net bound without changing the lattice point count."""
    if name not in _MOPS or m < 2 or q < 1:
        raise ValueError("Expected DTLZ1-4, M>=2, q>=1")
    return float(np.sqrt(m) / (4*q) if name == "DTLZ1" else m/(2*q))


def improved_coverage_interval(front: np.ndarray, name: str, *, q: int,
                               max_points: int = 100_000) -> dict:
    """Keep original net lower bound, replace only proven epsilon upper."""
    original = legacy_interval(front, name, q=q, max_points=max_points)
    epsilon = dependent_rounding_epsilon(name, original["M"], q)
    old_epsilon = legacy_epsilon(name, original["M"], q)
    if epsilon > old_epsilon + 1e-15:
        raise AssertionError("Refined certificate became looser")
    return {
        **original,
        "analytic_net_epsilon":epsilon,
        "continuous_radius_upper":original["observed_net_max_lower"]+
                                  epsilon+1e-12,
        "interval_width":epsilon+1e-12,
        "previous_interval_width":old_epsilon+1e-12,
        "refinement":"dependent rounding, no additional lattice points",
        "guarantee":"real arithmetic mathematical bound; FP not interval verified",
    }


def comparison(name: str, m: int, n: int, *, q_values: tuple[int,...],
               candidate_pool: int, seed: int, max_points: int) -> dict:
    if name not in ("DTLZ1","DTLZ2") or m < 3 or n < m or candidate_pool < n:
        raise ValueError("Expected DTLZ1/2, M>=3, n>=M, candidate pool>=n")
    mop = _MOPS[name](M=m)
    incumbent = np.asarray(mop.pf(n_points=n), dtype=float)
    proposal = farthest_candidates(name,m,n,candidates=candidate_pool,seed=seed)
    x = analytical_preimages(mop,proposal)
    replay = np.asarray(mop.evaluation(x)["F"],dtype=float)
    if not np.allclose(replay,proposal,atol=1e-10,rtol=0):
        raise ValueError("Decision-space inverse mismatch")
    return {"problem":name,"M":m,"budget":n,"seed":seed,
            "pool":candidate_pool,
            "methods":{
                label:{str(q):improved_coverage_interval(
                    points,name,q=q,max_points=max_points) for q in q_values}
                for label,points in (("incumbent",incumbent),("fps",replay))
            }}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objectives",nargs="+",type=int,default=[3,5])
    p.add_argument("--budgets",nargs="+",type=int,default=[128,512])
    p.add_argument("--resolutions",nargs="+",type=int,default=[8,16,24])
    p.add_argument("--seeds",nargs="+",type=int,default=[101,202,303])
    p.add_argument("--pool",type=int,default=4096)
    p.add_argument("--max-points",type=int,default=100000)
    a=p.parse_args()
    records=[comparison(name,m,n,q_values=tuple(a.resolutions),
                        candidate_pool=a.pool,seed=seed,max_points=a.max_points)
             for name in ("DTLZ1","DTLZ2")
             for m in a.objectives for n in a.budgets for seed in a.seeds]
    print(json.dumps({"schema":"gt119_dependent_rounding_v1",
                      "guarantee":"real arithmetic, FP enclosure not formally certified",
                      "records":records},sort_keys=True,indent=2))


if __name__=="__main__":
    main()
