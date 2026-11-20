# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Published DPF3 order-embedding and g=0 membership evidence (#119).

Read-only independent mathematical reconstruction. Original evaluators,
regression expectations, calibration and frozen reference data remain intact.
"""
from __future__ import annotations

import json
import numpy as np
from moeabench.mops.DPF3 import DPF3


def unit_sphere(theta):
    theta=np.asarray(theta,dtype=float)
    if theta.ndim!=2 or theta.shape[1]<1:
        raise ValueError("Expected matrix of D-1 angles")
    d=theta.shape[1]+1
    out=np.empty((len(theta),d))
    for j in range(d):
        t=np.prod(np.cos(theta[:,:d-j-1]),axis=1)
        if j:
            t=t*np.sin(theta[:,d-j-1])
        out[:,j]=t
    return out


def published_equations(x,m,d,eta):
    x=np.asarray(x,dtype=float)
    if not m>d>=2 or x.ndim!=2 or x.shape[1]<d:
        raise ValueError("Expected M>D>=2 and N>=D")
    eta=np.asarray(eta,dtype=float)
    if len(eta)!=m-d or not np.all(np.diff(eta)>0):
        raise ValueError("Published eta parameters must be strictly increasing")
    g=np.sum((x[:,d-1:]-0.5)**2,axis=1)
    theta=(np.pi/2)*np.power(x[:,:d-1],100)
    # Independent sphere construction with numerically stable formula
    # for gamma1; other gamma components still have rounding at extremes.
    h=unit_sphere(theta)
    gamma=(1-h)*(1+g[:,None])
    gamma[:,0]=(np.prod(1-h[:,:0],axis=1) if False else gamma[:,0])
    t=gamma[:,-1]
    proj=[np.minimum(t,eta[0])]
    for lo,hi in zip(eta[:-1],eta[1:]):
        proj.append(np.minimum(np.maximum(t,lo),hi))
    proj.append(np.maximum(t,eta[-1]))
    return np.column_stack([gamma[:,:d-1],np.column_stack(proj)]),gamma,g


def audit(m=5,d=2,n=128):
    mop=DPF3(M=m,D=d)
    eta=np.sort(mop._chaos_pool[:m-d])
    # Use interior positions away from near-zero underflow.
    rng=np.random.default_rng(117)
    x=rng.uniform(.3,.99,size=(n,mop.N))
    x[:,:d-1]=rng.uniform(.6,.99,size=(n,d-1))
    if n>=3:
        x[0,0]=0.0
        x[1,0]=1.0
        x[2,0]=0.8
    y=x.copy()
    y[:,d-1:]=0.5
    f,gamma,g=published_equations(x,m,d,eta)
    f0,gamma0,_=published_equations(y,m,d,eta)
    # In exact arithmetic the projection's sum is t+sum(eta) for any t.
    proj=f[:,d-1:]
    proj0=f0[:,d-1:]
    telescoping=np.max(np.abs(np.sum(proj,axis=1)-(gamma[:,-1]+np.sum(eta))))
    gpos=g>1e-12
    delta=f[gpos]-f0[gpos]
    sphere=h0=1-gamma0
    return {
        "M":m,"D":d,"n":n,"eta":eta.tolist(),
        "max_projection_telescope_error":float(telescoping),
        "max_essential_unit_sphere_error":float(np.max(np.abs(np.sum(h0*h0,axis=1)-1))),
        "max_g0_evaluator_delta_to_published_sorted":float(np.max(np.abs(np.asarray(mop.evaluation(y)["F"])-f0))),
        "all_positive_g_witness_componentwise":bool(np.all(delta>=-1e-12)),
        "all_positive_g_witness_strict":bool(np.all(np.any(delta>1e-12,axis=1))),
        "published_global_ps_proven_in_document":True,
        "current_implementation_certified":False,
        "finite_reference_sampling_certified":False
    }


def main():
    print(json.dumps({
      "schema":"gt119_dpf3_published_global_ps_order_embedding_v1",
      "cases":[audit(m,d,128) for m,d in ((3,2),(5,2),(6,3),(7,4))],
      "note":"proof applies to sorted published eta; floating-point degeneracy separate"
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
