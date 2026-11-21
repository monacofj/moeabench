# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent fixed-budget simplex candidates for DTLZ1 and DPF1 (#119).

Both problems share an explicitly identical DTLZ1 essential-simplex
parameterization. We use that proven mathematical equivalence to share
only a pure inverse mapping and a deterministic objective-space greedy
farthest-point selection over an independent uniform-simplex pool.

This is read-only research: no public MOP.ps()/pf(), test fixture,
calibration default, ADR or historical regression is changed.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from moeabench.mops.DTLZ1 import DTLZ1
from moeabench.mops.DPF1 import DPF1


def simplex_to_positions(fundamental):
    """Invert the standard DTLZ1 mapping for nonnegative simplex weights.

    Fundamental coordinates sum to one; the resulting objective simplex
    is 0.5 * fundamental. Axis vertices are handled with an explicit
    0/0 convention that preserves their exact objective images.
    """
    p=np.asarray(fundamental,dtype=float)
    if p.ndim!=2 or p.shape[1]<2 or np.any(p<0):
        raise ValueError("Expected matrix of nonnegative simplex coordinates")
    if np.max(np.abs(p.sum(axis=1)-1))>5e-13:
        raise ValueError("Simplex directions must sum to one")
    n,q=p.shape
    cumulative=np.cumsum(p,axis=1)
    x=np.empty((n,q-1),dtype=float)
    for j in range(q-1):
        numerator=cumulative[:,q-j-2]
        denominator=cumulative[:,q-j-1]
        x[:,j]=np.divide(numerator,denominator,
                         out=np.zeros_like(numerator),where=denominator>0)
    return x


def _objective_image(mop,essential):
    if isinstance(mop,DTLZ1):
        return essential
    d=mop.D
    w=mop._chaos_pool[:(mop.M-d)*d].reshape(mop.M-d,d)
    return np.column_stack([essential,essential@w.T])


def _sample(mop, directions):
    q=directions.shape[1]
    x=np.full((len(directions),mop.N),0.5)
    x[:,:q-1]=simplex_to_positions(directions)
    return x


def _farthest_objective_sample(points, count, anchors):
    if len(points)<count:
        raise ValueError("Candidate pool smaller than budget")
    chosen=list(range(anchors))
    distances=cKDTree(points[:anchors]).query(points)[0]
    for i in range(anchors,count):
        idx=int(np.argmax(distances))
        chosen.append(idx)
        new_distance=np.linalg.norm(points-points[idx],axis=1)
        distances=np.minimum(distances,new_distance)
    return np.asarray(chosen,dtype=int)


def candidate(mop,n,seed=11923,pool_multiplier=8):
    if n<2 or pool_multiplier<2:
        raise ValueError("Expected n>=2 and multiplier>=2")
    q=mop.M if isinstance(mop,DTLZ1) else mop.D
    rng=np.random.default_rng(seed)
    directions=np.vstack([
        np.eye(q),
        rng.dirichlet(np.ones(q),size=max(8*n,q)),
    ])
    fpool=_objective_image(mop,0.5*directions)
    selected=_farthest_objective_sample(fpool,n,min(q,n))
    ps=_sample(mop,directions[selected])
    evaluated=mop.evaluation(ps)["F"]
    expected=fpool[selected]
    error=float(np.max(np.abs(evaluated-expected)))
    if error>3e-12:
        raise AssertionError(f"Inverse simplex mapping objective mismatch {error}")
    if len(np.unique(evaluated,axis=0))!=n:
        raise AssertionError("Candidate did not generate distinct objectives")
    return ps,evaluated,error


def metrics(mop,gt,reference,axes):
    dist=cKDTree(gt).query(reference,k=1)[0]
    nn=cKDTree(gt).query(gt,k=2)[0][:,1]
    extrema=cKDTree(gt).query(axes,k=1)[0]
    return {
        "reference_probe_mean":float(np.mean(dist)),
        "reference_probe_p95":float(np.percentile(dist,95)),
        "reference_probe_max":float(np.max(dist)),
        "nearest_neighbor_cv":float(np.std(nn)/np.mean(nn)),
        "extreme_max_distance":float(np.max(extrema)),
    }


def compare(name,m,d,n,seed=11923,probes=2048):
    mop=DTLZ1(M=m) if name=="DTLZ1" else DPF1(M=m,D=d)
    q=m if name=="DTLZ1" else d
    rng=np.random.default_rng(seed+101)
    expected=0.5*rng.dirichlet(np.ones(q),size=probes)
    ref=_objective_image(mop,expected)
    axis=_objective_image(mop,0.5*np.eye(q))
    start=time.perf_counter()
    incumbent=mop.pf(n_points=n)
    incumbent_time=time.perf_counter()-start
    start=time.perf_counter()
    ps,proposed,error=candidate(mop,n,seed=seed)
    proposed_time=time.perf_counter()-start
    return {
        "name":name,"M":m,"D":d if name=="DPF1" else None,
        "n":n,"sampling_seed":seed,"independent_reference_seed":seed+101,"probe_count":probes,
        "incumbent":metrics(mop,incumbent,ref,axis),
        "objective_greedy_candidate":metrics(mop,proposed,ref,axis),
        "candidate_source_vs_analytic_max_abs":error,
        "incumbent_elapsed_seconds_observed":incumbent_time,
        "candidate_elapsed_seconds_observed":proposed_time,
        "candidate_proven_continuous_coverage":False,
        "changes_to_production":False,
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budgets",type=int,nargs="+",default=[128,512])
    ap.add_argument("--probes",type=int,default=2048)
    ap.add_argument("--seeds",type=int,nargs="+",default=[11923])
    opts=ap.parse_args()
    config=[
        ("DTLZ1",3,None),("DTLZ1",5,None),("DTLZ1",10,None),
        ("DPF1",3,2),("DPF1",5,2),("DPF1",5,3),
        ("DPF1",10,3),("DPF1",10,5),
    ]
    records=[compare(name,m,d,n,seed=seed,probes=opts.probes)
             for name,m,d in config for n in opts.budgets
             for seed in opts.seeds]
    print(json.dumps({
        "schema":"gt119_simplex_candidate_v1",
        "purpose":"fixed-budget independent coverage experiment, not a GT migration",
        "records":records,
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
