# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent audit of a corrected DPF5 finite sampler's angular cover.

Probes and published equations come from issue #119 reference reconstruction,
not the production evaluation function being checked. The grid theorem is
for exact real arithmetic; this reports sampled float64 correspondence.
"""
from __future__ import annotations

from itertools import product
import json

import numpy as np
from scipy.spatial import cKDTree

from moeabench.mops.DPF5 import DPF5
from scripts.audit.gt_dpf5_general_d_issue119 import reference
from scripts.audit.gt_dpf5_finite_sampler_issue119 import probes


def audit(m, d, requested, *, probe_count=256):
    mop=DPF5(M=m,D=d)
    decision=mop.ps(n_points=requested)
    evaluated=mop.evaluation(decision)["F"]
    expected=reference(decision,m,d)
    error=float(np.max(np.abs(evaluated-expected)))
    if error>5e-13:
        raise AssertionError(f"Published evaluator differs from independent oracle: {error}")
    if len(decision)!=requested or len(np.unique(evaluated,axis=0))!=requested:
        raise AssertionError("Returned DPF5 objective budget is not unique")
    holdout=probes(mop,probe_count,201)
    reports={}
    for branch in ("P","D"):
        selected=(decision[:,0]<1/3) if branch=="P" else (decision[:,0]>=1/3)
        points=evaluated[selected]
        q=m-1 if branch=="P" else d-1
        lo,hi=((0.0,np.nextafter(1/3,0.0)) if branch=="P"
               else (1/3,1.0))
        tree=cKDTree(points)
        completed_k=None
        k=2
        while k**q <= requested:
            angles=np.asarray(list(product(np.linspace(0,1,k),repeat=q)),
                              dtype=float).reshape(-1,q)
            angles[:,0]=lo+(hi-lo)*angles[:,0]
            x=np.full((len(angles),mop.N),0.5)
            x[:,:q]=angles
            x[:,m-1]=x[:,0]
            formula=reference(x,m,d)
            if np.max(tree.query(formula,k=1)[0]) > 5e-13:
                break
            completed_k=k
            k=2*k-1
        sample_probe=reference(holdout[branch],m,d)
        near=tree.query(sample_probe,k=1)[0]
        bound=None
        if completed_k is not None:
            bound=(np.pi/(4*(completed_k-1)))*np.sqrt((hi-lo)**2+q-1)
            if branch=="P":
                bound+=((1/3)-hi)*np.pi/2
            if np.max(near) > bound+1e-12:
                raise AssertionError("Observed probe violates conditional mathematical bound")
        reports[branch]={
            "objective_count":int(len(points)),
            "intrinsic_angular_dimension":int(q),
            "largest_completed_tensor_grid_k":completed_k,
            "ideal_real_angular_cover_bound":None if bound is None else float(bound),
            "independent_probe_max":float(np.max(near)),
            "independent_probe_median":float(np.median(near)),
            "full_continuous_float64_coverage_certified":False,
        }
    return {
        "M":int(m),"D":int(d),"requested":int(requested),
        "returned":len(decision),"global_numerical_unique":len(np.unique(evaluated,axis=0)),
        "max_production_vs_independent_published":error,
        "branches":reports,
        "conditional_exact_real_cover_bound":(
            max(v["ideal_real_angular_cover_bound"] for v in reports.values())
            if all(v["ideal_real_angular_cover_bound"] is not None for v in reports.values())
            else None
        ),
        "automatic_calibration_budget_chosen":False,
    }


def main():
    cases=[audit(m,d,n,probe_count=512)
           for m,d in ((3,2),(5,2),(6,3),(6,4))
           for n in (128,512)]
    print(json.dumps({"schema":"gt119_dpf5_corrected_conditional_cover_v1",
                      "scope":"real arithmetic bound and independent finite probes",
                      "cases":cases},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
