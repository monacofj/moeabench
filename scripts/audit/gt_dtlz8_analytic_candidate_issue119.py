# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only candidate for the exact constrained DTLZ8 Pareto front (#119).

A complete tensor grid in each analytic two-dimensional arm, together
with a complete grid on the central curve, gives a conservative
Euclidean continuous covering guarantee in *real arithmetic*.

This is independent research; it does not replace DTLZ8.ps()/pf(),
its frozen historical references, ADRs, or release regression tests.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from moeabench.mops.DTLZ8 import DTLZ8
from scripts.audit.gt_remaining_mop_matrix_issue119 import (
    dtlz8_same_t_pareto_witness,
)


def _central(t, m):
    t=np.asarray(t, dtype=float).reshape(-1)
    earlier=(1-t)/4
    return np.column_stack((np.repeat(earlier[:,None],m-1,axis=1),t))


def _arm(t, s, m, branch):
    """Published Pareto arm, at t in [0,1/3], s in [0,1]."""
    t,s=np.broadcast_arrays(np.asarray(t,dtype=float),np.asarray(s,dtype=float))
    t,s=t.ravel(),s.ravel()
    alpha=(1-t)/4
    beta=1-2*t
    small=alpha+s*(beta/2-alpha)
    large=beta-small
    f=np.column_stack((np.repeat(large[:,None],m-1,axis=1),t))
    f[:,branch]=small
    return f


def continuous_real_bound(m, central_count, t_count, s_count):
    """Exact-real-arithmetic covering bound for complete component grids.

    For the center: ||dF/dt|| = sqrt(1+(M-1)/16) and t-range=2/3.
    For each arm: sup ||dF/dt|| = sqrt((49*M-81)/16), while
    sup ||dF/ds|| = sqrt(M-1)/4, with t-range=1/3, s-range=1.
    Nearest complete-grid parameters differ by at most half a step.
    """
    if m < 3 or min(central_count,t_count,s_count)<2:
        raise ValueError("Need M>=3 and at least 2 samples per angular coordinate")
    center=np.sqrt(1+(m-1)/16)/(3*(central_count-1))
    arm=(np.sqrt((49*m-81)/16)/(6*(t_count-1))
         + np.sqrt(m-1)/(8*(s_count-1)))
    return {"center":float(center),"arms":float(arm),
            "union":float(max(center,arm))}


def choose_complete_grids(m,n):
    """Minimize the analytic worst-component covering bound at budget n.

    Nominal rows: central k_c and (M-1) independent k_t*k_s
    arm-grid rows. The shared boundary is duplicated at some points;
    extra budget after exact-unique filtering is filled separately.
    """
    minimum=2+4*(m-1)
    if m<3 or n<minimum:
        raise ValueError(f"Complete DTLZ8 component grids require n >= {minimum}")
    cap=(n-2)//(m-1)
    best=None
    for kt in range(2,cap//2+1):
        for ks in range(2,cap//kt+1):
            kc=n-(m-1)*kt*ks
            if kc<2:continue
            bound=continuous_real_bound(m,kc,kt,ks)
            rank=(bound["union"],bound["arms"],bound["center"],-kt,-ks)
            if best is None or rank<best[0]:
                best=(rank,kc,kt,ks,bound)
    if best is None:
        raise AssertionError("No complete grid split found")
    _,kc,kt,ks,bound=best
    return (kc,kt,ks),bound


def candidate(m,n,*,seed=11908):
    """Return n distinct evaluated objectives and valid PS preimages.

    The complete center and arm grids always remain in the sample.
    Deterministic Sobol replenishment adds objective-distinct points
    after merging mathematically repeated branch boundaries.
    """
    (kc,kt,ks),bound=choose_complete_grids(m,n)
    mop=DTLZ8(M=m)
    if mop.N % m:
        raise ValueError("This source-mapped candidate requires N divisible by M")
    saved,keys=[],set()

    def ingest(f):
        if len(saved)>=n:return
        f=np.atleast_2d(np.asarray(f,dtype=float))
        for row in f:
            key=tuple(float(y) for y in row)
            if key not in keys:
                keys.add(key)
                saved.append(row)
                if len(saved)>=n:break

    ingest(_central(np.linspace(1/3,1,kc),m))
    t,s=np.meshgrid(np.linspace(0,1/3,kt),np.linspace(0,1,ks),
                    indexing="ij")
    for branch in range(m-1):
        ingest(_arm(t,s,m,branch))

    attempt=0
    while len(saved)<n and attempt<8:
        needed=n-len(saved)
        for branch in range(m):
            if len(saved)>=n:break
            count=max(2, 2*needed)
            power=int(np.ceil(np.log2(count)))
            engine=qmc.Sobol(d=2,scramble=True,
                             seed=seed+attempt*m+branch)
            u=engine.random_base2(power)
            if branch==m-1:
                ingest(_central(1/3+(2/3)*u[:,0],m))
            else:
                ingest(_arm(u[:,0]/3,u[:,1],m,branch))
        attempt+=1
    if len(saved)!=n:
        raise RuntimeError("Could not generate objective-unique DTLZ8 Pareto points")

    exact=np.asarray(saved,dtype=float)
    decisions=np.repeat(exact,mop.N//m,axis=1)
    evaluated=np.asarray(mop.evaluation(decisions)["F"],dtype=float)
    valid=mop.evaluation(decisions,n_ieq_constr=mop.get_n_ieq_constr())
    reproject=dtlz8_same_t_pareto_witness(evaluated)
    if len(np.unique(evaluated,axis=0))!=n:
        raise AssertionError("Objective rounding introduced duplicate DTLZ8 points")
    if np.max(np.abs(evaluated-reproject))>1e-12:
        raise AssertionError("Analytic candidate is not on its published Pareto slice")
    if np.max(np.maximum(np.asarray(valid["G"]),0))>1e-12:
        raise AssertionError("Analytic candidate violated original DTLZ8 constraints")
    return {
        "mop":mop,"decisions":decisions,"objectives":evaluated,
        "grid_counts":{"central":kc,"arm_t":kt,"arm_s":ks,
                       "nominal":kc+(m-1)*kt*ks},
        "real_arithmetic_cover_bound":bound,
        "replay_max_abs":float(np.max(np.abs(evaluated-exact))),
    }


def analyze(m,n,*,probe_count=1024):
    data=candidate(m,n)
    rng=np.random.default_rng(7751)
    mop=data["mop"]
    ref=data["objectives"]
    tree=cKDTree(ref)
    components={}
    probes=_central(1/3+(2/3)*rng.random(probe_count),m)
    center=float(np.max(tree.query(probes)[0]))
    components["center"]=center
    for j in range(m-1):
        pt=_arm(rng.random(probe_count)/3,
                rng.random(probe_count),m,j)
        components[f"arm_{j+1}"]=float(np.max(tree.query(pt)[0]))
    observed=max(components.values())
    limit=data["real_arithmetic_cover_bound"]["union"]
    if observed>limit+1e-10:
        raise AssertionError("Finite validation probes exceeded ideal bound")
    return {
        "M":m,"requested":n,"returned":len(ref),
        "numerical_unique":len(np.unique(ref,axis=0)),
        "complete_grid_counts":data["grid_counts"],
        "mathematical_real_cover_upper":data["real_arithmetic_cover_bound"],
        "independent_probe_component_maxima":components,
        "independent_probe_global_max":observed,
        "objective_replay_max_abs":data["replay_max_abs"],
        "ieee_roundoff_enclosure_proved":False,
        "runtime_MOP_generator_modified":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objectives",type=int,nargs="+",default=[3,5,10])
    p.add_argument("--budgets",type=int,nargs="+",default=[128,512])
    p.add_argument("--probe-count",type=int,default=512)
    a=p.parse_args()
    records=[analyze(m,n,probe_count=a.probe_count)
             for m in a.objectives for n in a.budgets]
    print(json.dumps({"schema":"gt119_dtlz8_analytic_candidate_v1",
                      "scope":"exact mathematical Pareto model; float64 not interval certified",
                      "cases":records},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
