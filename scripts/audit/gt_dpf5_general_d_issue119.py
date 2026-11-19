# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent DPF5 published beta/gamma reconstruction for general M>D>=2.

Read-only comparison against original MOP evaluation; g=0 and g>0.
Published eq. (19)-(20): beta first M-D+1 spherical components;
gamma first spherical component repeated M-D+1 times with sqrt weight,
plus essential spherical tail gamma_2..gamma_D.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from moeabench.mops.DPF5 import DPF5


def spherical(theta: np.ndarray) -> np.ndarray:
    """Standard reversed-order positive hypersphere, one angle per M-1."""
    theta=np.asarray(theta, dtype=float)
    if theta.ndim != 2 or theta.shape[1]<1:
        raise ValueError("Need matrix with at least one angle")
    m=theta.shape[1]+1
    f=np.empty((len(theta),m),dtype=float)
    for i in range(m):
        val=np.prod(np.cos(theta[:,:m-i-1]),axis=1)
        if i:
            val=val*np.sin(theta[:,m-i-1])
        f[:,i]=val
    return f


def reference(x: np.ndarray,m: int,d: int) -> np.ndarray:
    x=np.asarray(x,dtype=float)
    if not (m>d>=2) or x.ndim!=2 or x.shape[1]<m:
        raise ValueError("Expected x 2D with N>=M>D>=2")
    theta=x[:,:m-1]*(np.pi/2)
    beta=spherical(theta)
    essential=spherical(theta[:,:d-1])
    repeated=m-d+1
    gamma=np.column_stack((
        np.repeat((essential[:,0:1]/np.sqrt(repeated)),repeated,axis=1),
        essential[:,1:]
    ))
    branch_d=x[:,0]>=1/3
    g=(x[:,m-1]-x[:,0])**2
    if x.shape[1]>m:
        g+=np.sum((x[:,m:]-0.5)**2,axis=1)
    return np.where(branch_d[:,None],gamma,beta)*(1+g[:,None])


def check(m: int,d: int,n: int=128,g_offset:float=0.0) -> dict:
    if m<=d or d<2 or n<4 or g_offset<0 or g_offset>0.2:
        raise ValueError("Unsupported dimension or study input")
    mop=DPF5(M=m,D=d)
    # Avoid class's sampler: independent interior PS with all
    # angular parameters active and reproducible without RNG.
    x=np.full((n,mop.N),0.5)
    x[:,0]=np.linspace(0.05,0.95,n)
    for j in range(1,m-1):
        x[:,j]=0.12+0.73*((np.arange(n)*(j+2)+j*11)% (n+1))/(n+1)
    x[:,m-1]=x[:,0]+np.where(x[:,0]<0.5,g_offset,-g_offset)
    actual=np.asarray(mop.evaluation(x)["F"],dtype=float)
    reconstructed=reference(x,m,d)
    g=(x[:,m-1]-x[:,0])**2
    results={"M":m,"D":d,"n":n,"g_offset":g_offset,"max_g":float(g.max())}
    for name,mask in (("P",x[:,0]<1/3),("D",x[:,0]>1/3)):
        a=actual[mask]
        b=reconstructed[mask]
        results[name]={
            "count":int(mask.sum()),
            "max_objective_difference":float(np.max(np.abs(a-b))),
            "actual_scaled_sphere_residual":float(np.max(np.abs(np.sum(a*a,axis=1)/(1+g[mask])**2-1))),
            "independent_scaled_sphere_residual":float(np.max(np.abs(np.sum(b*b,axis=1)/(1+g[mask])**2-1))),
        }
    results["global_pareto_completeness_certified"]=False
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--points",type=int,default=128)
    args=p.parse_args()
    cases=[check(m,d,args.points,g) for d,m in ((2,3),(2,5),(3,4),(3,5),(3,6),(3,8),(4,5),(4,6),(4,8))
           for g in (0,0.2)]
    print(json.dumps({"status":"independent published-equation reconstruction only; global PF not certified",
                      "cases":cases},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
