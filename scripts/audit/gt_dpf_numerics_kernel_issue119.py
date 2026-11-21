# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only independent DPF3 precision and DPF5 spherical-kernel probes.

Compares actual evaluator with equations on g=0 analytical PS; no MOP
source, published reference, old test or numerical baseline is changed.
Neither result alone establishes the globally complete Pareto front.
"""
from __future__ import annotations

import json
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from moeabench.mops.DPF3 import DPF3
from moeabench.mops.DPF5 import DPF5


def nd_count(f):
    return len(NonDominatedSorting().do(f, only_non_dominated_front=True))


def dpf3_precision(budget=512, m=3):
    if m < 3 or budget < 2:
        raise ValueError("M>=3, budget>=2")
    mop = DPF3(M=m, D=2)
    ps = np.asarray(mop.ps(n_points=budget))
    raw = np.asarray(mop.evaluation(ps)["F"])
    theta = np.float64(np.pi/2) * np.power(ps[:, 0], 100)
    direct = 1.0 - np.cos(theta)
    stable = 2 * np.sin(theta / 2)**2
    # Replace only the first essential objective; keep other coordinates
    # from the actual evaluator to isolate cancellation in 1-cos(theta).
    alternative = raw.copy()
    alternative[:, 0] = stable
    return {
        "M":m,"requested":budget,
        "raw_nondominated":nd_count(raw),
        "stable_first_component_nondominated":nd_count(alternative),
        "direct_zero_count":int(np.count_nonzero(direct == 0)),
        "stable_zero_count":int(np.count_nonzero(stable == 0)),
        "direct_zero_but_stable_positive":int(np.count_nonzero((direct == 0)&(stable > 0))),
        "max_abs_change_first_component":float(np.max(np.abs(stable-direct))),
        "zero_not_a_global_PF_certificate":True,
        "note":"stable first coordinate alone does not restore lost information in other rounded objectives",
    }


def canonical_dpf5_pf_from_ps(ps, m, d=2):
    """Equation (20), independent spherical beta/gamma, on g=0 PS only."""
    ps=np.asarray(ps,dtype=float)
    if d != 2 or m < 3 or ps.ndim != 2 or ps.shape[1] < m:
        raise ValueError("Expected M>=3, D=2, and N>=M")
    theta=(np.pi/2)*ps[:,:m-1]
    beta=np.empty((len(ps),m))
    for i in range(m):
        c = np.prod(np.cos(theta[:,:m-i-1]),axis=1)
        if i:
            c = c*np.sin(theta[:,m-i-1])
        beta[:,i]=c
    gamma=np.zeros((len(ps),m))
    gamma[:,:m-1]=np.cos(theta[:,0:1])/np.sqrt(m-1)
    gamma[:,m-1]=np.sin(theta[:,0])
    branch_d=ps[:,0] >= 1/3  # published first branch is x1 < 1/3
    return np.where(branch_d[:,None],gamma,beta)


def dpf5_kernel(budget=512,m=5):
    mop=DPF5(M=m,D=2)
    ps=np.asarray(mop.ps(n_points=budget))
    actual=np.asarray(mop.evaluation(ps)["F"])
    reference=canonical_dpf5_pf_from_ps(ps,m)
    p=ps[:,0] < 1/3
    d=ps[:,0] > 1/3
    def summarize(mask):
        if not np.any(mask): return {"count":0}
        values=actual[mask]
        canonical=reference[mask]
        return {
            "count":int(np.sum(mask)),
            "actual_max_unit_sphere_residual":float(np.max(np.abs(np.sum(values**2,axis=1)-1))),
            "canonical_max_unit_sphere_residual":float(np.max(np.abs(np.sum(canonical**2,axis=1)-1))),
            "actual_vs_canonical_max_abs_component":float(np.max(np.abs(values-canonical))),
            "actual_internal_dominated":int(len(values)-nd_count(values)),
            "canonical_internal_dominated":int(len(canonical)-nd_count(canonical)),
        }
    return {
        "M":m,"requested":budget,"returned":len(ps),
        "branch_P":summarize(p),"branch_D":summarize(d),
        "actual_union_dominated":int(len(actual)-nd_count(actual)),
        "canonical_union_dominated":int(len(reference)-nd_count(reference)),
        "published_threshold_equality_in_P":False,
        "global_pareto_completeness_certified":False,
    }


def main():
    print(json.dumps({
        "schema":"gt119_dpf_numerics_kernel_counterfactual_v1",
        "dpf3":[dpf3_precision(n,m) for m in (3,5) for n in (128,512)],
        "dpf5":[dpf5_kernel(n,m) for m in (3,5) for n in (128,512)],
        "status":"diagnostic only; no production generator changes",
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
