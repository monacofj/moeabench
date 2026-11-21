# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only, reproducible GT inventory for built-in MOPs outside DPF3/5 (#119).

This is an audit, not a GT calibration and not a proof of global Pareto
optimality. No production module, persistence sidecar, historical test,
numerical baseline, ADR or random global generator is modified.

The DTLZ5/6 complete-front theorem and corrective implementation are owned
by issue #65, not this workstream.

From repository root:
    python -m scripts.audit.gt_remaining_mops_issue119 --budgets 128 512 \
        --objectives 3 5 --intrinsic 2 3
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.spatial import cKDTree

from moeabench.mops import (
    DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ7, DTLZ8, DTLZ9,
    DPF1, DPF2, DPF4,
)
from moeabench.mops._dtlz7_pareto_regions import dtlz7_record_intervals

_MOPS = {
    "DTLZ1": DTLZ1, "DTLZ2": DTLZ2, "DTLZ3": DTLZ3,
    "DTLZ4": DTLZ4, "DTLZ7": DTLZ7, "DTLZ8": DTLZ8,
    "DTLZ9": DTLZ9, "DPF1": DPF1, "DPF2": DPF2, "DPF4": DPF4,
}
_SPHERICAL = {"DTLZ2", "DTLZ3", "DTLZ4"}
_CONSTRAINED = {"DTLZ8", "DTLZ9"}
_RECORD = {"DTLZ7", "DPF2"}


def _independent_reference(name: str, m: int, count: int, seed: int):
    """Independent objective-space laws for known full PFs (not MOP.ps())."""
    rng = np.random.default_rng(seed)
    if name == "DTLZ1":
        samples = rng.exponential(size=(count, m))
        return 0.5 * samples / samples.sum(axis=1, keepdims=True)
    if name in _SPHERICAL:
        samples = np.abs(rng.standard_normal(size=(count, m)))
        return samples / np.linalg.norm(samples, axis=1, keepdims=True)
    return None


def _record_occupancy(parameters: np.ndarray):
    (a, b), (c, d) = dtlz7_record_intervals()
    low = (parameters >= a - 1e-12) & (parameters <= b + 1e-12)
    high = (parameters > c) & (parameters <= d + 1e-12)
    valid = np.all(low | high, axis=1)
    labels = np.sum(high.astype(int) * (1 << np.arange(parameters.shape[1])), axis=1)
    values, counts = np.unique(labels[valid], return_counts=True)
    return {
        "valid_record_decisions": int(np.sum(valid)),
        "outside_record_domain": int(np.sum(~valid)),
        "possible_components": 2 ** parameters.shape[1],
        "sampled_components": len(values),
        "minimum_samples_to_touch_every_component": 2 ** parameters.shape[1],
        "all_components_touched_by_sample": bool(len(values)==2**parameters.shape[1]),
        "component_counts": {str(int(k)): int(v) for k, v in zip(values, counts)},
        "component_occupancy_is_not_a_coverage_certificate": True,
    }


def audit(name: str, m: int, n: int, *, d: int = 2, probe_count: int = 1024,
          seed: int = 11921) -> dict:
    """Measure the incumbent MOP contract and only *necessary* GT properties.

    A returning sample can be nondominated internally and still be globally
    dominated by a feasible point outside the sample. Finite probes similarly
    do not certify complete continuous objective-space coverage.
    """
    if name not in _MOPS or m < 2 or n < 2 or probe_count < 2:
        raise ValueError("Supported MOP, M>=2, n>=2, probe_count>=2 required")
    if name.startswith("DPF") and not (2 <= d < m):
        raise ValueError("DPF requires 2<=D<M")

    kwargs = {"M": m}
    if name.startswith("DPF"):
        kwargs["D"] = d
    mop = _MOPS[name](**kwargs)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        start = time.perf_counter()
        ps = np.asarray(mop.ps(n_points=n), dtype=float)
        pf = np.asarray(mop.pf(n_points=n), dtype=float)
        elapsed = time.perf_counter() - start

    if ps.ndim != 2 or ps.shape[1] != mop.N:
        raise ValueError(f"{name}: malformed PS with shape {ps.shape}")
    if pf.ndim != 2 or pf.shape[1] != m:
        raise ValueError(f"{name}: malformed PF with shape {pf.shape}")
    if not np.all(np.isfinite(ps)) or not np.all(np.isfinite(pf)):
        raise ValueError(f"{name}: generated nonfinite PS/PF")

    evaluated = np.asarray(mop.evaluation(ps)["F"], dtype=float)
    if evaluated.shape != (len(ps), m) or not np.all(np.isfinite(evaluated)):
        raise ValueError(f"{name}: malformed objective evaluator")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps_replay = np.asarray(mop.ps(n_points=n), dtype=float)
        pf_replay = np.asarray(mop.pf(n_points=n), dtype=float)
    exact_replay = bool(np.array_equal(ps, ps_replay) and
                        np.array_equal(pf, pf_replay))
    unique_pf = np.unique(pf, axis=0)
    nd_indices = NonDominatedSorting().do(unique_pf, only_non_dominated_front=True)
    nn = (cKDTree(unique_pf).query(unique_pf, k=2)[0][:, 1]
          if len(unique_pf) >= 2 else np.empty(0))
    nn_mean = float(np.mean(nn)) if len(nn) else None
    row = {
        "name": name, "M": m, "D": d if name.startswith("DPF") else None,
        "N": int(mop.N), "requested": n, "PS_rows": int(len(ps)),
        "PF_rows": int(len(pf)), "PF_float64_exact_unique": int(len(unique_pf)),
        "PF_unique_internal_dominated": int(len(unique_pf) - len(nd_indices)),
        "PF_duplicate_rows": int(len(pf) - len(unique_pf)),
        "PS_outside_box": int(np.sum(np.any((ps < -1e-12) | (ps > 1+1e-12),axis=1))),
        "PS_PF_evaluation_disagreement_max": (
            float(np.max(np.abs(pf-evaluated))) if len(pf)==len(ps) else None),
        "repeat_exact": exact_replay, "elapsed_seconds_observed": elapsed,
        "nearest_neighbor_mean": nn_mean,
        "nearest_neighbor_cv": (
            float(np.std(nn) / nn_mean) if nn_mean is not None and nn_mean > 0
            else None),
        "warnings": [str(w.message) for w in emitted],
        "certifies_global_Pareto_membership": False,
        "certifies_continuous_coverage": False,
    }

    if name == "DTLZ1":
        residual = pf.sum(axis=1) - 0.5
        row["simplex_sum_max_error"] = float(np.max(np.abs(residual)))
        row["simplex_extreme_max_gap"] = float(
            np.max(cKDTree(pf).query(0.5*np.eye(m))[0]))
    if name in _SPHERICAL:
        row["unit_sphere_norm_max_error"] = float(
            np.max(np.abs(np.linalg.norm(pf,axis=1)-1)))
        row["axis_extreme_max_gap"] = float(
            np.max(cKDTree(pf).query(np.eye(m))[0]))
    if name == "DPF1":
        # The first D dimensions are the essential DTLZ1-like simplex.
        essential = evaluated[:, :d]
        row["essential_simplex_sum_max_error"] = float(
            np.max(np.abs(essential.sum(axis=1)-0.5)))
        row["essential_simplex_extreme_max_gap"] = float(
            np.max(cKDTree(essential).query(0.5*np.eye(d))[0]))
    if name in _RECORD:
        q = m-1 if name == "DTLZ7" else d-1
        row["record_domain"] = _record_occupancy(ps[:, :q])
        trailing_start = m-1 if name == "DTLZ7" else d-1
        row["record_distance_variables_max_abs"] = float(
            np.max(np.abs(ps[:, trailing_start:])))
    if name in _CONSTRAINED:
        constrained = mop.evaluation(ps, n_ieq_constr=mop.get_n_ieq_constr())
        g = np.asarray(constrained["G"],dtype=float)
        row["constraints"] = {
            "G_shape": list(g.shape), "positive_G_max": float(np.maximum(g,0).max()),
            "infeasible_strict_float": int(np.sum(~constrained["feasible"])),
            "infeasible_above_1e-12": int(np.sum(np.max(g,axis=1)>1e-12)),
            "constraint_feasibility_is_not_global_Pareto_membership": True,
        }
    if name == "DPF4":
        # Do not substitute DPF3's sorted-eta rule without evidence that
        # the published DPF4 equation has the same projection contract.
        row["raw_chaos_thresholds_increasing"] = bool(
            np.all(np.diff(mop._chaos_pool[:m-d]) > 0))
        row["squared_projection_requires_paper_verification"] = True

    reference = _independent_reference(name,m,probe_count,seed)
    if reference is not None:
        ref_to_pf = cKDTree(pf).query(reference)[0]
        pf_to_ref = cKDTree(reference).query(pf)[0]
        row["independent_objective_probe"] = {
            "seed":seed,"n":probe_count,
            "reference_to_PF_median":float(np.median(ref_to_pf)),
            "reference_to_PF_p95":float(np.percentile(ref_to_pf,95)),
            "reference_to_PF_max_observed":float(ref_to_pf.max()),
            "PF_to_reference_max_observed":float(pf_to_ref.max()),
            "finite_probes_are_not_upper_bounds":True,
        }

    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names",nargs="+",choices=tuple(_MOPS),default=list(_MOPS))
    parser.add_argument("--objectives",nargs="+",type=int,default=[3,5])
    parser.add_argument("--budgets",nargs="+",type=int,default=[128,512])
    parser.add_argument("--intrinsic",nargs="+",type=int,default=[2,3])
    parser.add_argument("--probes",type=int,default=1024)
    parser.add_argument("--seed",type=int,default=11921)
    a=parser.parse_args()
    rows=[]
    for name in a.names:
        for m in a.objectives:
            for d in (a.intrinsic if name.startswith("DPF") else [None]):
                if d is not None and d>=m:
                    continue
                for n in a.budgets:
                    row=audit(name,m,n,d=d if d is not None else 2,
                              probe_count=a.probes,seed=a.seed)
                    rows.append(row)
    print(json.dumps({
        "schema":"gt119_remaining_mops_inventory_v1",
        "purpose":"diagnostic only; no GT or historical baseline replacement",
        "scope":"DTLZ1-4,DTLZ7-9,DPF1/2/4; DTLZ5/6 owned by issue 65",
        "records":rows,
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
