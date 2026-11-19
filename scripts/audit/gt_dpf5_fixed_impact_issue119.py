# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""DPF5 controlled fixed-decisions / fixed-objectives GT impact (#119).

Read-only: uses current production evaluator and independently reconstructed
published DPF5 equations on IDENTICAL decision vectors. Separates
(1) evaluator effect at fixed X and (2) metric reference effect at
bitwise fixed objective array. Current/counterfactual references are
finite samples; neither is certified globally Pareto-complete.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

from moeabench.mops.DPF5 import DPF5
from moeabench.metrics.GEN_igd import GEN_igd
from moeabench.metrics.GEN_igdplus import GEN_igdplus
from moeabench.metrics.GEN_hypervolume import GEN_hypervolume
from scripts.audit.gt_dpf5_general_d_issue119 import reference


def decision_matrix(mop, n, seed, *, off_front=False):
    if n < 8:
        raise ValueError("n must be at least 8")
    rng=np.random.default_rng(seed)
    x=np.full((n,mop.N),0.5,dtype=float)
    x[:,:mop.M-1]=rng.uniform(0.04,0.96,size=(n,mop.M-1))
    x[:,0]=np.linspace(0.04,0.96,n)
    x[:,mop.M-1]=x[:,0]
    if off_front:
        x[:,mop.M-1]=x[:,0]+np.where(x[:,0]<0.5,0.1,-0.1)
    return x


def _metrics(pop,gt,m):
    norm_min=np.zeros(m)
    norm_max=np.full(m,2.0)
    igd=float(GEN_igd([pop],gt).evaluate()[0])
    igdplus=float(GEN_igdplus([pop],gt).evaluate()[0])
    h_sol=float(GEN_hypervolume([pop],m,norm_min,norm_max).evaluate()[0])
    h_gt=float(GEN_hypervolume([gt],m,norm_min,norm_max).evaluate()[0])
    if not np.isfinite([igd,igdplus,h_sol,h_gt]).all() or h_gt<=0:
        raise ValueError("Unusable diagnostic metrics")
    return {"igd":igd,"igdplus":igdplus,"h_sol":h_sol,
            "h_gt":h_gt,"h_absolute_ratio":h_sol/h_gt}


def audit(m=5,d=2,n=64,population=16,seed=101):
    if m<=d or d<2 or population<8 or n<population:
        raise ValueError("Invalid dimensions or budgets")
    mop=DPF5(M=m,D=d)
    x_gt=decision_matrix(mop,n,seed)
    old_gt=np.asarray(mop.evaluation(x_gt)["F"],dtype=float)
    new_gt=reference(x_gt,m,d)
    x_pop=decision_matrix(mop,population,seed+1000,off_front=True)
    old_pop=np.asarray(mop.evaluation(x_pop)["F"],dtype=np.float64)
    new_pop=reference(x_pop,m,d)
    # Keep the objective population unchanged while changing only GT.
    fixed=old_pop.copy()
    digest=hashlib.sha256(fixed.tobytes()).hexdigest()
    old_metrics=_metrics(fixed,old_gt,m)
    new_metrics=_metrics(fixed,new_gt,m)
    np.testing.assert_array_equal(fixed,old_pop)
    assert hashlib.sha256(fixed.tobytes()).hexdigest()==digest
    assert old_metrics["h_sol"]==new_metrics["h_sol"]
    return {
        "M":m,"D":d,"requested_gt":n,"fixed_decisions_population":population,
        "seed":seed,"fixed_population_sha256":digest,
        "gt_max_component_change":float(np.max(np.abs(old_gt-new_gt))),
        "population_max_component_change_at_fixed_X":float(np.max(np.abs(old_pop-new_pop))),
        "current_reference":old_metrics,"published_formula_counterfactual":new_metrics,
        "metric_delta_counterfactual_minus_current":{
            k:float(new_metrics[k]-old_metrics[k]) for k in ("igd","igdplus","h_gt","h_absolute_ratio")},
        "norm_igd_note":"GEN_igd/igdplus use pymoo zero_to_one using each supplied GT; per-GT normalization may differ",
        "hv_note":"GEN_hypervolume fixed ideal=0, nadir=2, normalized reference=1.1 for all comparisons",
        "gt_pareto_membership_certified":False,
        "gt_pareto_completeness_certified":False,
    }


def main():
    print(json.dumps({
        "schema":"gt119_dpf5_fixed_x_fixed_population_metric_impact_v1",
        "status":"exploratory not GT correctness or release migration certification",
        "cases":[audit(m,d,64,16,seed)
                 for m,d in ((3,2),(5,2),(6,3),(6,4)) for seed in (101,202)],
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
