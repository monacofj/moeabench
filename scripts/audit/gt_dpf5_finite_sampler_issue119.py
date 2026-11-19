# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only finite DPF5 sampler coverage and effective-cardinality audit (#119).

Preserves branch membership in decision space. Compares the current PS
under the published *counterfactual* evaluator to independent stratified
angular probes; coverage measurements are finite diagnostics and use no
normative branch-mixture weights. Never changes production or frozen GT.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from scipy.spatial import cKDTree
from moeabench.mops.DPF5 import DPF5
from scripts.audit.gt_dpf5_general_d_issue119 import reference


def branch(x):
    """Published boundary convention: P strictly below 1/3, D otherwise."""
    return x[:, 0] >= (1.0 / 3.0)


def ps_from_angles(mop, angular):
    out = np.full((len(angular), mop.N), 0.5)
    out[:, :mop.M - 1] = angular
    out[:, mop.M - 1] = angular[:, 0]
    return out


def probes(mop, count, seed):
    """Independent branch-stratified interior probes, not sampled by PS."""
    rng = np.random.default_rng(seed)
    results = {}
    for key, lo, hi in (("P", 0.0, 1/3), ("D", 1/3, 1.0)):
        angles = rng.random((count, mop.M-1))
        angles[:,0] = lo+(hi-lo)*angles[:,0]
        results[key] = ps_from_angles(mop, angles)
    return results


def stats(m, d, requested=128, probe_count=512, seed=3101):
    if not (m>d>=2) or requested<8 or probe_count<32:
        raise ValueError("M>D>=2, requested>=8, probes>=32")
    mop = DPF5(M=m,D=d)
    ps = np.asarray(mop.ps(n_points=requested), dtype=float)
    if ps.ndim != 2 or ps.shape[1] != mop.N:
        raise ValueError("Unexpected PS shape")
    evaluated = np.asarray(mop.evaluation(ps)["F"], dtype=float)
    published = reference(ps,m,d)
    mask = branch(ps)
    sample_counts = {}
    coverage = {}
    unique_total = len(np.unique(published,axis=0))
    for key, keep in (("P", ~mask), ("D", mask)):
        sample = published[keep]
        # For an empty branch coverage is undefined, not zero.
        unique = np.unique(sample,axis=0)
        sample_counts[key] = {
            "decision_count": int(np.sum(keep)),
            "published_objective_unique_exact": int(len(unique)),
            "published_objective_duplicates_exact": int(len(sample)-len(unique)),
            "current_evaluator_objective_unique_exact": int(len(np.unique(evaluated[keep],axis=0))),
        }
        reference_probe = reference(probes(mop,probe_count,seed)[key],m,d)
        if len(unique):
            distance = cKDTree(unique).query(reference_probe,k=1)[0]
            coverage[key] = {
                "probe_count":probe_count,
                "probe_to_sample_median":float(np.median(distance)),
                "probe_to_sample_p95":float(np.quantile(distance,0.95)),
                "probe_to_sample_max":float(np.max(distance)),
                "note":"finite independent random probe diagnostic, not supremum/Hausdorff or certified epsilon",
            }
        else:
            coverage[key] = {"probe_count":probe_count,"status":"absent branch",
                             "probe_to_sample_max":None}
    extrema = {}
    # This checks coordinate-wise zero/one landmarks in objective space;
    # it does not certify component boundaries or Hausdorff coverage.
    for i in range(m):
        extrema[str(i)] = {
            "finite_sample_min":float(np.min(published[:,i])),
            "finite_sample_max":float(np.max(published[:,i])),
            "contains_zero_within_1e_12":bool(np.any(np.abs(published[:,i])<1e-12)),
            "contains_one_within_1e_12":bool(np.any(np.abs(published[:,i]-1)<1e-12)),
        }
    return {
        "M":m,"intrinsic_D":d,"requested":requested,"returned_ps":len(ps),
        "published_objective_unique_exact":unique_total,
        "current_evaluator_objective_unique_exact":int(len(np.unique(evaluated,axis=0))),
        "max_current_vs_published_component":float(np.max(np.abs(evaluated-published))),
        "branch_counts":sample_counts,
        "conditional_branch_coverage":coverage,
        "objective_coordinate_extrema":extrema,
        "global_membership_of_published_points":True,
        "finite_support_coverage_certified":False,
        "sampling_measure_selected":False,
        "note":"no automatic deduplication of published GT and no normative P/D mixing weight",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--probes",type=int,default=512)
    a=p.parse_args()
    rows = [stats(m,d,b,a.probes) for m,d in ((3,2),(5,2),(6,3),(6,4))
            for b in (128,512)]
    print(json.dumps({"schema":"gt119_dpf5_finite_sampler_coverage_v1",
                      "status":"observational; no GT promotion or branch-mixture law",
                      "cases":rows}, indent=2,sort_keys=True))


if __name__=="__main__":
    main()
