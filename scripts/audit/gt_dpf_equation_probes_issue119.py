# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only equation probes for published DPF3 and DPF5 contracts (#119).

A side-by-side counterfactual is NOT a production fix or global PF proof.
"""
from __future__ import annotations

import json
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from moeabench.mops.DPF3 import DPF3
from moeabench.mops.DPF5 import DPF5


def _nd(values):
    return len(NonDominatedSorting().do(values, only_non_dominated_front=True))


def dpf3_threshold_probe(m=5, n=512, d=2):
    mop = DPF3(M=m, D=d)
    ps = np.asarray(mop.ps(n_points=n))
    actual = np.asarray(mop.evaluation(ps)["F"])
    g = np.sum((ps[:, d - 1:] - 0.5)**2, axis=1)
    y = mop.eval_base_functions(ps, g, d)
    eta = np.asarray(mop._chaos_pool[:m-d], dtype=float)
    sorted_eta = np.sort(eta)
    last = y[:, d-1]
    projected = [np.minimum(last, sorted_eta[0])]
    for i in range(len(sorted_eta)-1):
        projected.append(np.minimum(np.maximum(last, sorted_eta[i]), sorted_eta[i+1]))
    projected.append(np.maximum(last, sorted_eta[-1]))
    prescribed = np.column_stack([y[:, :d-1], np.column_stack(projected)])
    return {
        "M": m, "D": d, "n": n,
        "eta_in_source_order": eta.tolist(),
        "eta_sorted": sorted_eta.tolist(),
        "source_order_strictly_increasing": bool(np.all(np.diff(eta) > 0)),
        "actual_vs_sorted_max_abs": float(np.max(np.abs(actual-prescribed))),
        "actual_internally_dominated": int(len(actual)-_nd(actual)),
        "sorted_counterfactual_internally_dominated": int(len(prescribed)-_nd(prescribed)),
        "global_pareto_membership_certified": False,
    }


def dpf5_threshold_probe(m=5, d=2, epsilon=1e-8):
    mop = DPF5(M=m, D=d)
    center = np.full((1, mop.N), 0.5)
    center[0, :m-1] = 0.4
    results = {}
    for key, x1 in (("left", 1/3-epsilon), ("equality", 1/3),
                    ("right", 1/3+epsilon)):
        x = center.copy()
        x[0, 0] = x1
        x[0, m-1] = x1  # preserve g=0
        results[key] = mop.evaluation(x)["F"][0].tolist()
    return {
        "M": m, "D": d, "epsilon": epsilon,
        "values": results,
        "equality_minus_left_max_abs": float(np.max(np.abs(np.asarray(results["equality"])-results["left"]))),
        "equality_minus_right_max_abs": float(np.max(np.abs(np.asarray(results["equality"])-results["right"]))),
        "published_at_equality": "gamma/D",
        "current_at_equality": "beta/P",
        "global_pareto_membership_certified": False,
    }


def main():
    print(json.dumps({
        "status":"independent counterfactual; production untouched",
        "dpf3":[dpf3_threshold_probe(m) for m in (3,4,5,10)],
        "dpf5":[dpf5_threshold_probe(m) for m in (3,5)],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
