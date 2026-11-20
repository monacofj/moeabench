# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only DPF5 branch-wise fixed-unique-budget sampler comparison (#119).

The branch allocation is COPIED from the incumbent (not optimized or adopted).
Comparison: current incumbent distinct objectives vs a seeded scrambled Sobol
counterfactual with deterministic branch-boundary / corner anchors.
All distances are independent finite probes, not certified Hausdorff bounds.
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from moeabench.mops.DPF5 import DPF5
from scripts.audit.gt_dpf5_general_d_issue119 import reference
from scripts.audit.gt_dpf5_finite_sampler_issue119 import ps_from_angles, probes, branch


def sobol_angles(m: int, d: int, key: str, total: int, seed: int):
    """Generate candidate angles; D draws only D-1 essential parameters."""
    dim = m - 1 if key == "P" else d - 1
    lower, upper = (0.0, 1 / 3) if key == "P" else (1 / 3, 1.0)
    engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
    power = max(1, int(np.ceil(np.log2(max(2, total * 3)))))
    raw = engine.random_base2(power)
    angles = np.full((len(raw), m - 1), 0.5)
    angles[:, :dim] = raw
    angles[:, 0] = lower + (upper - lower) * raw[:, 0]
    return angles


def anchors(m: int, d: int, key: str):
    """Finite deterministic boundary/corner landmarks, not exhaustive extremes."""
    dim = m - 1 if key == "P" else d - 1
    arr = []
    for first in ((0.0, np.nextafter(1 / 3, 0.0)) if key == "P"
                  else (1 / 3, 1.0)):
        a = np.full(m - 1, 0.5)
        a[0] = first
        arr.append(a.copy())
        for j in range(1, dim):
            for v in (0.0, 1.0):
                b = a.copy()
                b[j] = v
                arr.append(b)
        # All corners / spherical coordinate axis extremes at first-angle endpoint.
        for v in (0.0, 1.0):
            b = a.copy()
            b[1:dim] = v
            arr.append(b)
    return np.asarray(arr)


def candidate(mop, key: str, target: int, seed: int):
    if target < 1:
        return np.empty((0, mop.M)), 0.0
    start = time.perf_counter()
    out, seen = [], set()
    # Preserve target number of *exact* distinct objectives; no global
    # deduplication mixing P and D probability masses.
    def ingest(angles):
        nonlocal out
        ps = ps_from_angles(mop, angles)
        values = reference(ps, mop.M, mop.D)
        for row in values:
            sig = tuple(float(v) for v in row)
            if sig not in seen:
                seen.add(sig)
                out.append(row)
                if len(out) == target:
                    break
    # Fixed anchors first; if target is very small all anchors cannot fit.
    ingest(anchors(mop.M, mop.D, key))
    for attempt in range(3):
        if len(out) >= target:
            break
        ingest(sobol_angles(mop.M, mop.D, key, max(16, target - len(out)), seed + attempt))
    if len(out) != target:
        raise ValueError("Could not produce requested distinct branch objectives")
    return np.asarray(out), time.perf_counter() - start


def compare(m=5, d=2, requested=512, *, seed=71, probe_seeds=(201, 202, 203), probes_per_branch=512):
    if not m > d >= 2 or requested < 32 or not probe_seeds:
        raise ValueError("Invalid M, D, budget or probe-seed list")
    mop=DPF5(M=m,D=d)
    ps=np.asarray(mop.ps(n_points=requested))
    evaluated=reference(ps,m,d)
    mask=branch(ps)
    reports={}
    for label, selection in (("P",~mask),("D",mask)):
        old=np.unique(evaluated[selection],axis=0)
        if len(old) == 0:
            reports[label]={"status":"absent incumbent branch"}
            continue
        new, elapsed=candidate(mop,label,len(old),seed)
        if len(new) != len(old) or len(np.unique(new,axis=0)) != len(new):
            raise AssertionError("Effective budget mismatch")
        distances=[]
        for probe_seed in probe_seeds:
            q=reference(probes(mop,probes_per_branch,probe_seed)[label],m,d)
            one={}
            for key, points in (("incumbent",old),("stratified_sobol",new)):
                dist=cKDTree(points).query(q,k=1)[0]
                one[key]={"median":float(np.median(dist)),
                          "p95":float(np.quantile(dist,0.95)),
                          "max_observed":float(dist.max())}
            distances.append({"probe_seed":int(probe_seed), "measurements":one})
        reports[label]={
            "effective_unique_budget":int(len(old)),
            "incumbent_decision_rows":int(selection.sum()),
            "candidate_generation_seconds_observed":float(elapsed),
            "probe_results":distances,
            "median_improvement_per_seed":[
                float(t["measurements"]["incumbent"]["median"] -
                      t["measurements"]["stratified_sobol"]["median"])
                for t in distances],
            "conditional_measure_not_selected":True
        }
    return {
        "M":m,"intrinsic_D":d,"requested":requested,
        "actual_decision_count":len(ps),
        "candidate_branch_allocation_is_incumbent":True,
        "seed":seed,"probe_seeds":list(probe_seeds),
        "probes_per_branch":probes_per_branch,
        "branches":reports,
        "certified_supremum_or_uniformity":False,
        "release_GT_approved":False
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--probes",type=int,default=512)
    a=p.parse_args()
    cases=[compare(m,d,requested,probes_per_branch=a.probes)
           for m,d in ((3,2),(5,2),(6,3),(6,4))
           for requested in (128,512)]
    print(json.dumps({
        "schema":"gt119_dpf5_same_effective_budget_branch_sobol_v1",
        "status":"finite exploratory comparison; no sampler replacement approved",
        "cases":cases
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
