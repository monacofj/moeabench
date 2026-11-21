# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit component-stratified GT candidates for DTLZ7/DPF2 (#119).

The exact Pareto-record parameter domain has 2^q disconnected components,
q=M-1 for DTLZ7 and q=D-1 for DPF2. At n>=2^q the candidate touches
EVERY component; at n<2^q it occupies n distinct components (the maximum
possible) and explicitly cannot certify whole-front component presence.

This script does not change public samplers or any scientific baseline.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from moeabench.mops.DTLZ7 import DTLZ7
from moeabench.mops.DPF2 import DPF2
from moeabench.mops._dtlz7_pareto_regions import dtlz7_record_intervals


def _problem(name,m,d):
    if name=="DTLZ7":
        return DTLZ7(M=m)
    if name=="DPF2":
        return DPF2(M=m,D=d)
    raise ValueError("DTLZ7 or DPF2 required")


def _essential_dimensions(mop):
    return (mop.M-1) if isinstance(mop,DTLZ7) else (mop.D-1)


def _positions_from_labels(labels,q,seed):
    """Produce strictly record-optimal decision parameters per component."""
    n=len(labels)
    (lo0,hi0),(lo1,hi1)=dtlz7_record_intervals()
    # Published record interval 2 has an open lower endpoint; exclude
    # its potentially non-strict Pareto reentry boundary.
    lo1=max(float(np.nextafter(lo1,hi1)),lo1+1e-9)
    engine=qmc.Sobol(d=q,scramble=True,seed=seed)
    u=engine.random_base2(int(np.ceil(np.log2(max(2,n)))))[:n]
    labels=np.asarray(labels,dtype=np.int64)
    bits=((labels[:,None] >> np.arange(q,dtype=np.int64)) & 1).astype(bool)
    lo=np.where(bits,lo1,lo0)
    hi=np.where(bits,hi1,hi0)
    return lo+(hi-lo)*u


def candidate(mop,n,seed=11930):
    if not isinstance(n,(int,np.integer)) or n<1:
        raise ValueError("Expected positive integer count")
    q=_essential_dimensions(mop)
    if q>=30:
        raise ValueError("Too many components to enumerate safely in audit")
    components=2**q
    rng=np.random.default_rng(seed)
    if n<components:
        # Distinct components are mathematically the best possible
        # *occupancy* at a budget smaller than component count.
        labels=rng.choice(components,size=n,replace=False)
    else:
        labels=np.tile(np.arange(components,dtype=np.int64),n//components)
        rem=n%components
        if rem:
            labels=np.concatenate((labels,rng.permutation(components)[:rem]))
        labels=rng.permutation(labels)
    position=_positions_from_labels(labels,q,seed+101)
    decision=np.zeros((n,mop.N),dtype=float)
    decision[:,:q]=position
    # Both analytical models have optimal trailing decision variables 0.
    return decision,labels


def _distances(front,ref):
    d=cKDTree(front).query(ref,k=1)[0]
    return {"mean":float(np.mean(d)),
            "p95":float(np.percentile(d,95)),
            "max_observed":float(np.max(d))}


def compare(name,m,d,n,seed=11930,probes=2048):
    mop=_problem(name,m,d)
    q=_essential_dimensions(mop)
    total=2**q
    t0=time.perf_counter()
    incumbent=mop.ps(n_points=n)
    incumbent_F=mop.evaluation(incumbent)["F"]
    t1=time.perf_counter()
    candidate_X,labels=candidate(mop,n,seed)
    candidate_F=mop.evaluation(candidate_X)["F"]
    t2=time.perf_counter()
    if len(np.unique(candidate_F,axis=0))!=n:
        raise AssertionError("Candidate did not return n distinct objectives")
    reference_labels=np.random.default_rng(seed+887).integers(total,size=probes)
    reference_positions=_positions_from_labels(reference_labels,q,seed+999)
    reference_X=np.zeros((probes,mop.N),dtype=float)
    reference_X[:,:q]=reference_positions
    reference_F=mop.evaluation(reference_X)["F"]
    (a,b),(c,d1)=dtlz7_record_intervals()
    x=incumbent[:,:q]
    old_bits=(x>c).astype(np.int64)
    old_labels=np.sum(old_bits*(1<<np.arange(q)),axis=1)
    return {
        "name":name,"M":m,"D":d if name=="DPF2" else None,
        "sampling_seed":seed,"requested":n,"components_total":total,
        "budget_can_touch_all_components":bool(n>=total),
        "incumbent_component_count":int(len(np.unique(old_labels))),
        "candidate_component_count":int(len(np.unique(labels))),
        "candidate_max_possible_components":min(n,total),
        "incumbent":_distances(incumbent_F,reference_F),
        "candidate":_distances(candidate_F,reference_F),
        "incumbent_seconds_observed":t1-t0,
        "candidate_seconds_observed":t2-t1,
        "finite_probe_not_continuous_certificate":True,
        "public_MOP_mutated":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budgets",type=int,nargs="+",default=[128,512])
    p.add_argument("--probes",type=int,default=2048)
    p.add_argument("--seeds",type=int,nargs="+",default=[11930])
    a=p.parse_args()
    configs=[
        ("DTLZ7",3,None),("DTLZ7",5,None),("DTLZ7",10,None),
        ("DPF2",5,2),("DPF2",5,3),
        ("DPF2",10,5),("DPF2",10,9),
    ]
    records=[compare(name,m,d,n,seed=seed,probes=a.probes)
             for name,m,d in configs for n in a.budgets
             for seed in a.seeds]
    print(json.dumps({"schema":"gt119_record_component_stratification_candidate_v1",
                      "purpose":"fixed-budget component presence experiment, no baseline replacement",
                      "records":records},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
