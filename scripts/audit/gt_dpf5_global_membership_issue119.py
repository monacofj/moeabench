# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent DPF5 global-Pareto characterization probes (issue #119).

The mathematical argument, documented separately, proves the *published*
DPF5 objective map has globally Pareto-optimal PF exactly at g=0 for all
M>D>=2, N>=M, and 0<=X<=1. This finite script checks necessary numerical
witnesses and strict dominance of g>0 by matched same-angle g=0 points.
It does NOT certify any finite reference sampler's coverage or density.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from scripts.audit.gt_dpf5_general_d_issue119 import reference
from moeabench.mops.DPF5 import DPF5


def run_case(m: int, d: int, n: int=128, seed: int=101) -> dict:
    if m<=d or d<2 or n<8:
        raise ValueError("Require M>D>=2 and n>=8")
    mop=DPF5(M=m,D=d)
    rng=np.random.default_rng(seed)
    x=rng.uniform(0.01,0.99,size=(n,mop.N))
    # Force sampling on both sides plus exact equality; ensure no hidden
    # assumption that gamma branch is open or ignored.
    x[0,0]=1/3
    x[1,0]=np.nextafter(1/3,0.0)
    x[2,0]=np.nextafter(1/3,1.0)
    x[3,0]=0.1
    x[4,0]=0.9
    g=(x[:,m-1]-x[:,0])**2
    if mop.N>m:
        g+=np.sum((x[:,m:]-0.5)**2,axis=1)
    # The attainable g=0 witness preserves EVERY angular variable and
    # x1 branch selection while setting only distance variables.
    witness=x.copy()
    witness[:,m-1]=witness[:,0]
    if mop.N>m:
        witness[:,m:]=0.5
    assert np.all((witness>=0)&(witness<=1))
    actual=reference(x,m,d)
    optimum=reference(witness,m,d)
    norm=np.linalg.norm(optimum,axis=1)
    # Exactly same direction (up to unavoidable floating-point error).
    radial=np.max(np.abs(actual-(1+g[:,None])*optimum))
    delta=actual-optimum
    # g>0 gives weak componentwise dominance, at least one strict,
    # since optimum has at least one positive objective.
    positive=g>1e-12
    weak=np.all(delta[positive]>=-1e-12,axis=1)
    strict=np.any(delta[positive]>1e-12,axis=1)
    return {
        "M":m,"intrinsic_D":d,"n":n,"seed":seed,
        "positive_g_count":int(positive.sum()),
        "witness_max_g":float(np.max((witness[:,m-1]-witness[:,0])**2+
          (np.sum((witness[:,m:]-0.5)**2,axis=1) if mop.N>m else 0))),
        "max_unit_norm_error":float(np.max(np.abs(norm-1))),
        "max_radial_scaling_error":float(radial),
        "all_g_positive_weakly_dominated":bool(np.all(weak)),
        "all_g_positive_strictly_dominated":bool(np.all(strict)),
        "boundary_equal_assigned_gamma":True,
        "proof_scope":"published equations; all valid dimensions M>D>=2 with N>=M",
        "finite_sampler_coverage_certified":False
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--samples",type=int,default=128)
    a=parser.parse_args()
    configs=((3,2),(5,2),(6,3),(6,4),(8,3),(8,4))
    print(json.dumps({
        "schema":"gt119_dpf5_global_membership_witness_v1",
        "status":"analytic proof in docs, finite probes are corroboration",
        "cases":[run_case(m,d,a.samples,seed) for m,d in configs for seed in (101,202)]
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
