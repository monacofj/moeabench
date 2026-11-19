# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent high-precision DPF3 and cross-dimensional DPF5 diagnostics.

Research only: no change to production evaluators, reference GT or regressions.
High precision uses the *actual finite-float decision coordinates* as input;
it does not imply all sampled decisions are globally Pareto-optimal.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from moeabench.mops.DPF3 import DPF3
from moeabench.mops.DPF5 import DPF5
from scripts.audit.gt_dpf_numerics_kernel_issue119 import canonical_dpf5_pf_from_ps


def precision_probe(n=512, digits=100):
    if n < 2 or digits < 60:
        raise ValueError("Require n>=2 and >=60 decimal digits")
    import mpmath as mp  # optional research dependency; lazy import
    mop=DPF3(M=3,D=2)
    ps=np.asarray(mop.ps(n_points=n),dtype=float)
    theta=np.pi/2*np.power(ps[:,0],100)
    naive=1-np.cos(theta)
    stable=2*np.sin(theta/2)**2
    with mp.workdps(digits):
        halfpi=mp.pi/2
        # mp.mpf(float) preserves the exact binary input decision coordinate.
        exact=[2*mp.sin(halfpi*mp.mpf(float(x))**100/2)**2 for x in ps[:,0]]
        exact_positive=np.asarray([z>0 for z in exact],dtype=bool)
        rounded=np.asarray([float(z) for z in exact],dtype=float)
        reference_max=float(max(exact))
    mask=exact_positive&(naive==0)
    mask_stable=exact_positive&(stable==0)
    return {
        "requested":n,"mp_decimal_digits":digits,
        "mp_positive_count":int(np.sum(exact_positive)),
        "naive_zero_but_mp_positive":int(np.sum(mask)),
        "stable_zero_but_mp_positive":int(np.sum(mask_stable)),
        "mp_to_float64_zero_count":int(np.sum(rounded==0)),
        "stable_vs_mp_float64_max_abs":float(np.max(np.abs(stable-rounded))),
        "reference_max":reference_max,
        "warning":"high precision uses the stable trigonometric identity; zero after float64 conversion is representability loss",
    }


def dpf5_dimension_probe(m=5,n=256,d=2):
    if m<=d or d!=2:
        raise ValueError("Only published D=2 reconstruction currently supported")
    mop=DPF5(M=m,D=d)
    ps=np.asarray(mop.ps(n_points=n),dtype=float)
    runtime=np.asarray(mop.evaluation(ps)["F"],dtype=float)
    reference=canonical_dpf5_pf_from_ps(ps,m,d)
    # Compare norm residual separately for strict P and strict D,
    # respecting that the exact x1=1/3 boundary is a separate convention.
    valid=(ps[:,0]!=1/3)
    p=(ps[:,0]<1/3)&valid
    deg=(ps[:,0]>1/3)&valid
    result={"M":m,"D":d,"requested":n,"actual_count":len(ps)}
    for name,mask in (("P",p),("D",deg)):
        if not mask.any():
            result[name]={"count":0}
            continue
        observed=runtime[mask]; canonical=reference[mask]
        result[name]={
            "count":int(mask.sum()),
            "max_abs_component_difference":float(np.max(np.abs(observed-canonical))),
            "actual_norm2_minus_one_max_abs":float(np.max(np.abs(np.sum(observed**2,axis=1)-1))),
            "reference_norm2_minus_one_max_abs":float(np.max(np.abs(np.sum(canonical**2,axis=1)-1))),
        }
    result["globally_complete_certified"]=False
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits",type=int,default=160)
    parser.add_argument("--budgets",nargs="+",type=int,default=[128,512])
    parser.add_argument("--objectives",nargs="+",type=int,default=[3,4,5,6,10])
    opts=parser.parse_args()
    print(json.dumps({
        "status":"read-only counterfactual, not global Pareto proof",
        "dpf3":[precision_probe(n,opts.digits) for n in opts.budgets],
        "dpf5":[dpf5_dimension_probe(m,n) for m in opts.objectives for n in opts.budgets],
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
