# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only structural audit of complex MOP Pareto references, issue #119.

Actual MOP ps()/pf() and evaluation are used WITHOUT changing them.
Checks necessary properties: budget cardinality, finiteness, bounds,
feasibility, internal nondominance, and DTLZ7/DPF2 record-region
occupancy. These are NOT sufficient for global Pareto completeness.

DTLZ5/6 M>=4 g=0 subsets are explicitly incomplete (issue #65).
A finite sampled set with no internal dominance does not prove that
its points are globally Pareto-optimal or that all components are present.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from moeabench.mops.DTLZ5 import DTLZ5
from moeabench.mops.DTLZ6 import DTLZ6
from moeabench.mops.DTLZ7 import DTLZ7
from moeabench.mops.DTLZ8 import DTLZ8
from moeabench.mops.DTLZ9 import DTLZ9
from moeabench.mops.DPF2 import DPF2
from moeabench.mops.DPF3 import DPF3
from moeabench.mops.DPF4 import DPF4
from moeabench.mops.DPF5 import DPF5
from moeabench.mops._dtlz7_pareto_regions import dtlz7_record_intervals


_CLASSES = {
    "DTLZ5": DTLZ5, "DTLZ6": DTLZ6, "DTLZ7": DTLZ7,
    "DTLZ8": DTLZ8, "DTLZ9": DTLZ9,
    "DPF2": DPF2, "DPF3": DPF3, "DPF4": DPF4, "DPF5": DPF5,
}


def record_components(parameters: np.ndarray) -> dict:
    """Classify each position coordinate by independent record intervals.

    The two intervals arise from the analytical one-dimensional
    benefit function x(1+sin(3*pi*x)); the function checks membership
    and records observed combinations, without claiming continuous
    coverage inside each disconnected component.
    """
    p = np.asarray(parameters, dtype=float)
    if p.ndim != 2 or p.shape[1] < 1 or not np.isfinite(p).all():
        raise ValueError("Expected finite position parameter matrix")
    (a,b),(c,d)=dtlz7_record_intervals()
    in_first=(p >= a-1e-11)&(p <= b+1e-11)
    in_second=(p > c-1e-12)&(p <= d+1e-11)
    valid=np.all(in_first|in_second,axis=1)
    labels=(in_second.astype(np.int64) * (1 << np.arange(p.shape[1]))).sum(axis=1)
    counts={str(int(k)):int(v) for k,v in zip(*np.unique(labels[valid],return_counts=True))}
    return {
        "valid_parameter_rows":int(valid.sum()),
        "out_of_region_rows":int((~valid).sum()),
        "expected_component_count":int(2**p.shape[1]),
        "observed_component_count":len(counts),
        "component_counts":counts,
        "record_boundaries":[[a,b],[c,d]],
        "component_completeness_certified":False,
    }


def audit(name: str, m: int, budget: int, *, intrinsic_d: int = 2) -> dict:
    if name not in _CLASSES or m < 3 or budget < 1:
        raise ValueError("Supported complex MOP, M>=3, budget>=1")
    kwargs={"M":m}
    if name.startswith("DPF"):
        kwargs["D"]=intrinsic_d
        if not (2 <= intrinsic_d < m):
            raise ValueError("DPF intrinsic dimension requires 2<=D<M")
    mop=_CLASSES[name](**kwargs)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        decision=np.asarray(mop.ps(n_points=budget),dtype=float)
        front=np.asarray(mop.pf(n_points=budget),dtype=float)
    if decision.ndim!=2 or front.ndim!=2 or decision.shape[1]!=mop.N or front.shape[1]!=m:
        raise ValueError("Malformed PS/PF dimensions")
    if not (np.isfinite(decision).all() and np.isfinite(front).all()):
        raise ValueError("Non-finite PS/PF generated")
    evaluated=np.asarray(mop.evaluation(decision)["F"],dtype=float)
    if evaluated.shape!=(len(decision),m) or not np.isfinite(evaluated).all():
        raise ValueError("Malformed forward evaluation of PS")
    if len(decision)==len(front):
        replay=float(np.max(np.abs(front-evaluated)))
    else:
        replay=None  # DPF3 may internally remove dominated candidates
    ranks=NonDominatedSorting().do(front,only_non_dominated_front=True)
    duplicate=int(len(front)-len(np.unique(front,axis=0)))
    row={
        "name":name,"M":m,"D":intrinsic_d if name.startswith("DPF") else None,
        "N":mop.N,"requested_points":budget,
        "ps_returned":len(decision),"pf_returned":len(front),
        "pf_shortfall":budget-len(front),
        "pf_duplicates":duplicate,
        "pf_internal_dominated":len(front)-len(ranks),
        "decision_out_of_unit_box":int(np.sum(np.any(
            (decision < -1e-12)|(decision > 1+1e-12),axis=1))),
        "ps_to_pf_exact_order_max_difference":replay,
        "warnings":[str(item.message) for item in emitted],
        "globally_pareto_complete_certified":False,
    }
    if name in ("DTLZ5","DTLZ6"):
        # #65 replaces the historical incomplete g=0-only sampler for M>=4.
        # This finite audit still cannot *prove* continuous Pareto coverage.
        row["known_incomplete_subfront_for_M_ge_4"]=False
        row["numerically_bounded_full_front_sampler"]=(m >= 4)
        row["distance_g_from_generated_decision_variables"]={
            "DTLZ5":float(np.max(np.sum((decision[:,m-1:]-0.5)**2,axis=1))),
            "DTLZ6":float(np.max(np.sum(decision[:,m-1:]**0.1,axis=1))),
        }[name]
    if name in ("DTLZ7","DPF2"):
        position=decision[:,:m-1] if name=="DTLZ7" else decision[:,:intrinsic_d-1]
        row["record_domain"]=record_components(position)
    if name in ("DTLZ8","DTLZ9"):
        ev=mop.evaluation(decision,n_ieq_constr=mop.get_n_ieq_constr())
        violations=np.maximum(np.asarray(ev["G"],dtype=float),0)
        max_violation=np.max(violations,axis=1)
        row["constraints"]={
            "constraint_count":int(ev["G"].shape[1]),
            "infeasible_generated_PS_strict_float":int((~ev["feasible"]).sum()),
            "max_positive_G":float(np.max(violations)),
            "infeasible_generated_PS_tol_1e_12":int(np.sum(max_violation>1e-12)),
            "infeasible_generated_PS_tol_1e_8":int(np.sum(max_violation>1e-8)),
            "feasibility_is_not_pareto_optimality":True,
        }
    if name in ("DPF3","DPF4","DPF5"):
        raw_rank=NonDominatedSorting().do(evaluated,only_non_dominated_front=True)
        row["raw_ps_internal_dominated"]=len(evaluated)-len(raw_rank)
        row["pf_may_be_filtered"]=(name=="DPF3")
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names",nargs="+",choices=tuple(_CLASSES),default=list(_CLASSES))
    parser.add_argument("--objectives",nargs="+",type=int,default=[3,5])
    parser.add_argument("--budgets",nargs="+",type=int,default=[128,512])
    parser.add_argument("--dpf-intrinsic",type=int,default=2)
    a=parser.parse_args()
    records=[audit(name,m,n,intrinsic_d=a.dpf_intrinsic)
             for name in a.names for m in a.objectives for n in a.budgets]
    print(json.dumps({
        "schema":"gt119_complex_mop_necessary_invariants_v1",
        "status":"necessary-conditions-only; global Pareto completeness NOT certified",
        "records":records,
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
