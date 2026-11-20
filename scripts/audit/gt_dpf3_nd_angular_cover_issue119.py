# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""DPF3 arbitrary-D published-model angular-grid covering certificate (#119).

A spherical angular map is 1-Lipschitz in Euclidean angle space, as its
angular partial derivatives are orthogonal with norm <= 1. Complementing
and ordered scalar min/max projections cannot increase Euclidean distances.
Thus a tensor product k-grid (D-1 angles) gives Hausdorff bound
sqrt(D-1)*pi/[4(k-1)] for the continuous published PF, in REAL arithmetic.

The finite-precision error is measured separately using an mpmath angular
round-trip diagnostic. Measurements are not rigorous IEEE interval bounds.
No production MOP code or regression is changed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import numpy as np
import mpmath as mp
from scipy.spatial import cKDTree
from moeabench.mops.DPF3 import DPF3
from scripts.audit.gt_dpf3_global_membership_issue119 import published_equations


def published_from_theta(theta, eta):
    """Rebuild paper's DPF3 on g=0, bypassing the x^100 inverse entirely."""
    from scripts.audit.gt_dpf3_global_membership_issue119 import unit_sphere
    theta = np.asarray(theta, dtype=float)
    h = unit_sphere(theta)
    essential = 1 - h
    # cos-complement stable when first objective is small (D=2 etc.)
    # Do not assume all higher-D complement products are stable in float.
    t = essential[:, -1]
    projections = [np.minimum(t, eta[0])]
    for lo, hi in zip(eta[:-1],eta[1:]):
        projections.append(np.minimum(np.maximum(t, lo),hi))
    projections.append(np.maximum(t, eta[-1]))
    return np.column_stack([essential[:, :-1], np.column_stack(projections)])


def grid(k, intrinsic_d):
    if k < 2 or intrinsic_d < 2:
        raise ValueError("Need k,D>=2")
    angles=np.linspace(0,np.pi/2,k)
    return np.asarray(list(itertools.product(angles,repeat=intrinsic_d-1)),dtype=float)


def audit(m=5,d=3,k=7,probes=2048,seed=311):
    if not m>d>=2 or k<2 or probes<64 or k**(d-1)>100_000:
        raise ValueError("Unsupported M,D,k or probe budget")
    mop=DPF3(M=m,D=d)
    eta=np.sort(mop._chaos_pool[:m-d])
    theta=grid(k,d)
    exact_model_float=published_from_theta(theta,eta)
    mathematical_bound=float(np.sqrt(d-1)*np.pi/(4*(k-1)))
    rng=np.random.default_rng(seed)
    theta_probe=rng.uniform(0,np.pi/2,size=(probes,d-1))
    probes_F=published_from_theta(theta_probe,eta)
    observed=float(cKDTree(exact_model_float).query(probes_F,k=1)[0].max())
    if observed>mathematical_bound+2e-12:
        raise AssertionError("Sampled distance violates analytic bound")

    x=np.full((len(theta),mop.N),.5,dtype=float)
    x[:,:d-1]=np.power(theta/(np.pi/2),1/100)
    # The actual NUMERICAL decision coordinates are rounded. Reconstruct
    # their angles and published objective values without using production.
    evaluated_at_rounded_ps, _, _ = published_equations(x,m,d,eta)
    objective_error=float(np.linalg.norm(evaluated_at_rounded_ps-exact_model_float,axis=1).max())
    theta_round=(np.pi/2)*np.power(x[:,:d-1],100)
    angular_error=float(np.linalg.norm(theta_round-theta,axis=1).max())
    with mp.workdps(100):
        half=mp.pi/2
        highprecision_angle_error=max(
            mp.sqrt(mp.fsum([
                (half*mp.mpf(float(xi))**100-half*mp.mpf(int(idx))/(k-1))**2
                for xi,idx in zip(xrow,indexes)
            ]))
            for xrow,indexes in zip(
                x[:,:d-1],
                itertools.product(range(k),repeat=d-1)
            )
        )
    # Report under float64 because this is an empirical high-precision
    # estimate, not a certified interval enclosure.
    return {
        "M":m,"D":d,"grid_per_angle":k,
        "angular_dimensions":d-1,
        "requested_tensor_points":len(theta),
        "float_exact_unique_objectives":len(np.unique(exact_model_float,axis=0)),
        "analytic_real_Hausdorff_upper_bound":mathematical_bound,
        "independent_probe_count":probes,
        "observed_probe_max":observed,
        "float_roundtrip_angle_max_norm":angular_error,
        "high_precision_angle_roundtrip_max_norm_estimate":float(highprecision_angle_error),
        "float_published_objective_roundtrip_max_norm":objective_error,
        "theorem_for_published_continuous_PF":True,
        "rigorous_float_rounding_enclosure":False,
        "effective_budget_is_not_exact_objective_uniqueness":True,
        "calibration_default_approved":False
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--probes",type=int,default=1024)
    a=p.parse_args()
    rows=[audit(m,d,k,a.probes) for m,d,k in
          ((3,2,33),(5,2,129),(5,3,5),(5,3,9),(6,4,4),(6,4,7))]
    print(json.dumps({"schema":"gt119_dpf3_nd_cover_bound_v1",
                      "status":"real-arithmetic theorem; sampled FP errors not certified",
                      "cases":rows},sort_keys=True,indent=2))


if __name__=="__main__":
    main()
