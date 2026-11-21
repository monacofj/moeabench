# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent DPF5 D=2 nonzero-g evaluation / sphere-radius diagnostic.

No production mutation. This tests formula-evaluation agreement for
off-Pareto feasible decision points; it does not assert they belong to PF.
"""
from __future__ import annotations
import json
import numpy as np
from moeabench.mops.DPF5 import DPF5
from scripts.audit.gt_dpf_numerics_kernel_issue119 import canonical_dpf5_pf_from_ps


def audit(m: int, n: int = 128, offset: float = 0.2) -> dict:
    if m < 3 or n < 4 or not (0 < offset < 0.4):
        raise ValueError("Expected M>=3, n>=4, 0<offset<0.4")
    mop=DPF5(M=m,D=2)
    points=np.full((n,mop.N),0.5)
    # Explicit deterministic interior branch samples. x_M chosen off x_1;
    # all remaining distance variables stay at 0.5.
    x1=np.linspace(0.05,0.95,n)
    points[:,0]=x1
    points[:,1:m-1]=0.4
    points[:,m-1]=x1+np.where(x1<=0.5,offset,-offset)
    g=np.square(points[:,m-1]-x1)
    actual=np.asarray(mop.evaluation(points)["F"])
    predicted=canonical_dpf5_pf_from_ps(points,m)*(1+g[:,None])
    valid=np.abs(x1-1/3)>1e-12
    report={"M":m,"n":n,"g_min":float(np.min(g)),"g_max":float(np.max(g))}
    for key,mask in (("P",(x1<1/3)&valid),("D",(x1>1/3)&valid)):
        observed=actual[mask]
        baseline=predicted[mask]
        report[key]={
            "count":int(mask.sum()),
            "max_abs_objective_difference":float(np.max(np.abs(observed-baseline))),
            "actual_scaled_norm2_residual":float(np.max(np.abs(np.sum(observed**2,axis=1)/(1+g[mask])**2-1))),
            "reference_scaled_norm2_residual":float(np.max(np.abs(np.sum(baseline**2,axis=1)/(1+g[mask])**2-1))),
        }
    report["global_PF_optimality_certified"]=False
    return report


def main():
    print(json.dumps({
        "schema":"gt119_dpf5_off_pareto_d2_v1",
        "status":"evaluation equation only, g>0, not Pareto membership",
        "cases":[audit(m,n) for m in (3,4,5,6,10) for n in (128,512)],
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
