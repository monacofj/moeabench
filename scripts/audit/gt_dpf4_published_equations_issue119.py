# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent, read-only published DPF4 equation audit (issue #119).

Published Zhen et al. DPF4, Section IV-D/F: *increasingly sorted*
logistic eta thresholds and strictly increasing phi(t)=t**2 transforms.
The current BaseDPF chaotic sequence is intentionally left untouched
so DPF1/DPF2 retain their independent published meanings.

All comparisons use fixed decisions. No production MOP is modified.
"""
import json

import numpy as np

from moeabench.mops.DPF4 import DPF4


def _nonnegative_unit_hypersphere(angles):
    n, q = angles.shape
    c, s = np.cos(angles), np.sin(angles)
    h = np.empty((n, q + 1), dtype=float)
    for j in range(q + 1):
        count = q-j
        a = np.prod(c[:, :count], axis=1)
        if j:
            a *= s[:, count]
        h[:, j] = a
    return h


def published_dpf4(X: np.ndarray, M: int, D: int, raw_eta):
    """Independent published equation, not DPF4.eval_base_functions()."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    theta = X[:, :D-1] * (np.pi/2)
    h = _nonnegative_unit_hypersphere(theta)
    tail = X[:, D-1:] - 0.5
    g = 100 * (tail.shape[1] + np.sum(
        tail**2 - np.cos(20*np.pi*tail), axis=1))
    gamma = (1 + g[:, None])*(1-h)
    eta = np.sort(np.asarray(raw_eta, dtype=float)[:M-D])
    if M == D:
        return gamma**2
    t = gamma[:, -1]
    projected = [np.minimum(t, eta[0])]
    for lower, upper in zip(eta[:-1],eta[1:]):
        projected.append(np.minimum(np.maximum(t, lower),upper))
    projected.append(np.maximum(t, eta[-1]))
    return np.column_stack([gamma[:, :-1], *projected])**2


def audit(M: int, D: int, *, sample_count=256, seed=11942):
    mop=DPF4(M=M,D=D)
    rng=np.random.default_rng(seed)
    X=rng.uniform(0.05,0.95,size=(sample_count,mop.N))
    X[:, D-1:]=0.5
    actual=np.asarray(mop.evaluation(X)["F"])
    expected=published_dpf4(X,M,D,mop._chaos_pool)
    eta_raw=mop._chaos_pool[:M-D]
    t=(1-_nonnegative_unit_hypersphere(X[:,:D-1]*np.pi/2)[:,-1])
    source_projected=np.sqrt(np.maximum(actual[:,D-1:],0))
    telescoping_error=float(np.max(np.abs(
        source_projected.sum(axis=1)-(t+np.sum(np.sort(eta_raw))))))
    # Probe an extremely small but exactly feasible x1 in D=2. It
    # exposes the avoidable 1-cos cancellation after phi(z)=z**2.
    tiny=None
    if D==2:
        v=np.full((1,mop.N),0.5)
        v[0,0]=1e-10
        computed=float(mop.evaluation(v)["F"][0,0])
        angle=np.pi/2*v[0,0]
        stable_expected=float((2*np.sin(angle/2)**2)**2)
        tiny={"x1":1e-10,"current_first_objective":computed,
              "stable_published_first_objective":stable_expected,
              "positive_is_rounded_to_zero":computed==0 and stable_expected>0}
    return {
        "name":"DPF4","M":M,"D":D,"cases":sample_count,
        "eta_raw":eta_raw.tolist(),
        "eta_sorted":np.sort(eta_raw).tolist(),
        "raw_eta_strictly_increasing":bool(np.all(np.diff(eta_raw)>0)),
        "max_objective_difference_to_independent_published":float(
            np.max(np.abs(actual-expected))),
        "source_projection_telescope_error":telescoping_error,
        "small_angle_numerical_probe":tiny,
        "reference_global_Pareto_membership_proved_by_this_finite_audit":False,
        "no_production_changes":True,
    }


def main():
    cases=[audit(m,d) for m,d in [(3,2),(5,2),(6,3),(7,4)]]
    print(json.dumps({"schema":"gt119_dpf4_published_equations_v1",
                      "source":"Zhen et al. arXiv:1806.02706v2, IV-D/F",
                      "purpose":"independent diagnostic, no mutations",
                      "cases":cases},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
