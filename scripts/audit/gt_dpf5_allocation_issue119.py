# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only DPF5 total-effective-budget P/D allocation sensitivity (#119).

Fixed total unique-objective budget, independent sampler and probe seeds.
P/D allocations are explicit counterfactuals; no probability law or
calibration choice is adopted. Finite probes are NOT certified coverings.
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from moeabench.mops.DPF5 import DPF5
from scripts.audit.gt_dpf5_general_d_issue119 import reference
from scripts.audit.gt_dpf5_finite_sampler_issue119 import branch, probes
from scripts.audit.gt_dpf5_branch_sampler_compare_issue119 import candidate


def summarize(m=5, d=2, requested=512, *, sampler_seeds=(71,72,73),
              probe_seeds=(201,202,203), probe_count=256):
    if m<=d or d<2 or requested<32 or not sampler_seeds or not probe_seeds or probe_count<32:
        raise ValueError("Bad M/D, budget or seeds")
    mop=DPF5(M=m,D=d)
    ps=np.asarray(mop.ps(n_points=requested))
    published=reference(ps,m,d)
    is_d=branch(ps)
    old={"P":np.unique(published[~is_d],axis=0),
         "D":np.unique(published[is_d],axis=0)}
    if not all(len(x)>0 for x in old.values()):
        raise ValueError("Both branches must be represented")
    total=sum(map(len,old.values()))
    # Note that cross-branch exact overlap is not generally impossible;
    # the split total is an explicit STRATIFIED effective-count budget.
    # This study reports global unique counts separately.
    allocations={"incumbent":(len(old["P"]),len(old["D"])),
                 "half_P":(total//2,total-total//2),
                 "two_thirds_P":((2*total)//3,total-(2*total)//3)}
    probe_sets={key:[reference(probes(mop,probe_count,seed)[key],m,d)
                     for seed in probe_seeds] for key in ("P","D")}
    records=[]
    for label,(p_count,d_count) in allocations.items():
        for seed in sampler_seeds:
            start=time.perf_counter()
            if label=="incumbent":
                points={"P":old["P"],"D":old["D"]}
            else:
                points={key:candidate(mop,key,count,seed)[0]
                        for key,count in (("P",p_count),("D",d_count))}
            elapsed=time.perf_counter()-start
            if sum(map(len,points.values())) != total:
                raise AssertionError("Different total branchwise effective budgets")
            joined=np.concatenate([points["P"],points["D"]],axis=0)
            metrics={}
            for key in ("P","D"):
                seed_measures=[]
                for probe_seed,q in zip(probe_seeds,probe_sets[key]):
                    dd=cKDTree(points[key]).query(q,k=1)[0]
                    seed_measures.append({
                        "probe_seed":int(probe_seed),
                        "median":float(np.median(dd)),
                        "p95":float(np.quantile(dd,.95)),
                        "max_observed":float(np.max(dd))})
                metrics[key]={"points":len(points[key]),
                              "probe_measurements":seed_measures,
                              "mean_probe_median":float(np.mean([x["median"] for x in seed_measures])),
                              "worst_observed_probe_distance":float(max(x["max_observed"] for x in seed_measures))}
            # No weighted pooling over mixed-dimensional supports:
            # criterion here is *maximum conditional* observed distance.
            records.append({
                "allocation":label,"sampler_seed":int(seed),
                "branchwise_total":total,
                "global_exact_unique":int(len(np.unique(joined,axis=0))),
                "P_count":len(points["P"]),"D_count":len(points["D"]),
                "max_conditional_probe_distance":float(max(metrics["P"]["worst_observed_probe_distance"],
                                                           metrics["D"]["worst_observed_probe_distance"])),
                "generation_seconds_observed":float(elapsed),
                "branches":metrics})
    return {"M":m,"intrinsic_D":d,"requested":requested,
            "incumbent_returned_decisions":len(ps),
            "incumbent_global_unique_published":int(len(np.unique(published,axis=0))),
            "fixed_branchwise_total":total,
            "allocations":{name:{"P":int(v[0]),"D":int(v[1])}
                           for name,v in allocations.items()},
            "sampler_seeds":list(sampler_seeds),"probe_seeds":list(probe_seeds),
            "probe_count_each_branch":probe_count,
            "records":records,
            "normative_mixed_dimensional_measure_adopted":False,
            "calibration_default_authorized":False,
            "finite_probe_max_not_certified_worst_case":True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--probes",type=int,default=256)
    args=p.parse_args()
    rows=[summarize(m,d,b,probe_count=args.probes)
          for m,d in ((3,2),(5,2),(6,3),(6,4)) for b in (128,512)]
    print(json.dumps({"schema":"gt119_dpf5_allocation_sensitivity_v1",
                      "status":"exploratory, no branch distribution adopted",
                      "cases":rows},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
