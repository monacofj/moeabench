# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact mathematical DPF3 D=2 angular reference with proved covering bound.

Research-only reference construction using sorted published eta thresholds.
A nonnegative unit-speed 1D curve admits the certified Euclidean covering
radius pi/(4*(n-1)) for an n-point endpoint-inclusive uniform angular grid.
The bound is mathematical in real arithmetic; float64 realization is audited
but does not automatically inherit an exact rounding-free certificate.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from scipy.spatial import cKDTree
from moeabench.mops.DPF3 import DPF3


def published_curve(theta, eta):
    """Stable complement cos and ordered saturating projections for D=2."""
    t=np.asarray(theta,dtype=float)
    e=np.asarray(eta,dtype=float)
    if t.ndim!=1 or np.any((t<0)|(t>np.pi/2)) or e.ndim!=1 or len(e)<1 or np.any(np.diff(e)<=0):
        raise ValueError("Expected angles in [0, pi/2] and sorted eta")
    # gamma1=1-cos(theta) in stable form; t_last=1-sin(theta)
    first=2*np.sin(t/2)**2
    last=1-np.sin(t)
    projections=[np.minimum(last,e[0])]
    for lo,hi in zip(e[:-1],e[1:]):
        projections.append(np.minimum(np.maximum(last,lo),hi))
    projections.append(np.maximum(last,e[-1]))
    return np.column_stack((first,*projections))


def study(m=5,n=128,probes=4097):
    if m<3 or n<2 or probes<64:
        raise ValueError("Require M>=3, n>=2, probes>=64")
    mop=DPF3(M=m,D=2)
    eta=np.sort(np.asarray(mop._chaos_pool[:m-2],dtype=float))
    angles=np.linspace(0,np.pi/2,n)
    values=published_curve(angles,eta)
    if len(np.unique(values,axis=0))!=n:
        raise AssertionError("D=2 angular PF candidate has duplicate float64 outputs")
    dense_theta=np.linspace(0,np.pi/2,probes)
    dense=published_curve(dense_theta,eta)
    measured=cKDTree(values).query(dense,k=1)[0]
    bound=float(np.pi/(4*(n-1)))
    # Small allowance in finite precision (bound itself is real-arithmetic).
    if np.max(measured)>bound+2e-13:
        raise AssertionError("Measured DPF3 curve violates proven bound")
    x=np.full((n,mop.N),0.5)
    x[:,0]=np.power(angles/(np.pi/2),1/100)
    actual=np.asarray(mop.evaluation(x)["F"],dtype=float)
    # Actual is not guaranteed to equal published curve (unsorted thresholds,
    # and 1-cos(theta) loses resolution).
    return {
        "M":m,"D":2,"n":n,"probes":probes,
        "published_sorted_eta":eta.tolist(),
        "published_sample_unique_exact":len(np.unique(values,axis=0)),
        "certified_real_arithmetic_upper_bound":bound,
        "measured_probe_max":float(np.max(measured)),
        "measured_probe_p95":float(np.quantile(measured,0.95)),
        "actual_evaluator_vs_published_max_abs":float(np.max(np.abs(actual-values))),
        "incumbent_pf_count":int(len(mop.pf(n_points=n))),
        "real_arithmetic_coverage_proved":True,
        "floating_point_coverage_proved":False,
        "applies_to_general_D":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--probes",type=int,default=4097)
    a=p.parse_args()
    print(json.dumps({
        "schema":"gt119_dpf3_d2_certified_angular_cover_v1",
        "status":"read-only; exact real-arithmetic D=2 published-model bound",
        "cases":[study(m,n,a.probes) for m in (3,5,10) for n in (128,512)],
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
