# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent read-only remaining-MOP GT inventory for issue #119.

Audits actual analytical PS/PF generators for DTLZ1-4, DTLZ7-9 and
DPF1/2/4 without changing production, existing tests or reference data.
DTLZ5/6 full Pareto-front correctness is deliberately reserved for #65.

The checks are necessary diagnostics, NOT global Pareto completeness
proofs or finite continuous-Hausdorff certificates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.spatial import cKDTree

from moeabench.mops import (
    DPF1, DPF2, DPF4, DTLZ1, DTLZ2, DTLZ3, DTLZ4,
    DTLZ7, DTLZ8, DTLZ9,
)
from moeabench.mops._dtlz7_pareto_regions import dtlz7_record_intervals

CLASSES = {
    cls.__name__: cls for cls in (
        DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ7, DTLZ8, DTLZ9,
        DPF1, DPF2, DPF4
    )
}


def _record_component_count(parameters):
    """Count all observed disconnected DTLZ7-style Pareto-record patches."""
    (_, upper_0), (lower_1, _) = dtlz7_record_intervals()
    first = parameters <= upper_0 + 1e-11
    second = parameters >= lower_1 - 1e-12
    valid = first | second
    patterns = set(tuple(int(v) for v in row) for row in second[np.all(valid, axis=1)])
    return {
        "expected": int(2 ** parameters.shape[1]),
        "observed": len(patterns),
        "outside_record_domain": int(np.sum(~np.all(valid, axis=1))),
        "component_counts": [
            {
                "binary_label": "".join(str(int(v)) for v in key),
                "count": int(np.sum(np.all(second == key, axis=1))),
            }
            for key in itertools.product((False, True), repeat=parameters.shape[1])
        ],
    }


def _source_kind(name, m):
    if name == "DTLZ8" and m in (3, 5, 10):
        p = Path(__file__).resolve().parents[2] / "moeabench" / "mops" / "data" / f"DTLZ8_{m}_optimal_x.csv"
        if p.is_file():
            return "bundled_static_csv"
    if name == "DTLZ8":
        return "guided_analytical_fallback"
    if name in ("DTLZ2", "DTLZ3", "DTLZ4"):
        return "native_surface_uniform_direction_inverse"
    if name in ("DPF2", "DTLZ7"):
        return "native_disconnected_record_parameters"
    if name in ("DPF1", "DPF4"):
        return "shared_BaseDPF_parameter_sampler"
    return "native_problem_specific"


def _spacing(unique):
    if len(unique) < 2:
        return {"nearest_mean": None, "nearest_cv": None}
    d = cKDTree(unique).query(unique, k=2)[0][:, 1]
    mean = float(d.mean())
    return {
        "nearest_mean": mean,
        "nearest_cv": float(d.std() / mean) if mean > 0 else None,
    }



def dtlz8_same_t_pareto_witness(objectives):
    """Independently construct a feasible dominating witness at unchanged f_M.

    For t=f_M>=1/3, each earlier objective is minimized at
    alpha=(1-t)/4. For t<1/3, all pair sums must exceed
    beta=1-2*t and each coordinate must exceed alpha. Let a be the
    smallest earlier objective. If a<=beta/2, all other coordinates
    can be decreased to beta-a. Otherwise all can be reduced to
    beta/2. This gives the complete Pareto-minimal slice for fixed t.

    The construction is deterministic, operates in *objective* space,
    and does not rely on the MOP's PS sampler. It cannot prove a
    sampled objective belongs to the global PF without separately
    establishing cross-t nondominance, but it can prove strict
    domination whenever the witness improves a feasible sample.
    """
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    if f.ndim != 2 or f.shape[1] < 3:
        raise ValueError("DTLZ8 witness requires M>=3")
    t = f[:, -1]
    alpha = (1 - t) / 4
    beta = 1 - 2 * t
    witness = f.copy()
    high = t >= (1 / 3)
    witness[high, :-1] = alpha[high, None]
    for i in np.flatnonzero(~high):
        earlier = f[i, :-1]
        j = int(np.argmin(earlier))
        minimum = earlier[j]
        if minimum <= beta[i] / 2:
            witness[i, :-1] = beta[i] - minimum
            witness[i, j] = minimum
        else:
            witness[i, :-1] = beta[i] / 2
    return witness



def _published_dpf4_at_ps(decisions, m, d, eta):
    """Independent published DPF4 projection with *sorted* thresholds.

    The positive hypersphere is independently reconstructed at g=0;
    the squared min/max projections use the increasingly ordered
    logistic-map thresholds required by Zhen et al. (2022), Sec. IV-F.
    This is not a replacement production evaluator.
    """
    x = np.atleast_2d(np.asarray(decisions, dtype=float))
    q = d - 1
    theta = x[:, :q] * (np.pi / 2)
    c = np.cos(theta)
    sn = np.sin(theta)
    h = np.ones((len(x), d), dtype=float)
    for i in range(d):
        length = d - i - 1
        if length:
            h[:, i] *= np.prod(c[:, :length], axis=1)
        if i:
            h[:, i] *= sn[:, length]
    gdata = x[:, d-1:]
    g = 100 * (
        gdata.shape[1] +
        np.sum((gdata - .5)**2 - np.cos(20*np.pi*(gdata-.5)), axis=1)
    )
    y = (1 + g[:, None]) * (1 - h)
    thresholds = np.sort(np.asarray(eta, dtype=float))
    if m == d:
        return y**2
    t = y[:, -1]
    projections = [np.minimum(t, thresholds[0])]
    for a, b in zip(thresholds[:-1], thresholds[1:]):
        projections.append(np.minimum(np.maximum(t, a), b))
    projections.append(np.maximum(t, thresholds[-1]))
    return np.column_stack((y[:, :-1], np.column_stack(projections)))**2


def audit(name: str, m: int, budget: int, *, intrinsic_d: int = 2):
    if name not in CLASSES:
        raise ValueError(f"Unsupported audited MOP {name}")
    if not isinstance(budget, int) or budget < 2 or m < 2:
        raise ValueError("Need M >= 2 and requested points >= 2")
    is_dpf = name.startswith("DPF")
    if is_dpf and not 2 <= intrinsic_d < m:
        raise ValueError("DPF intrinsic dimension must satisfy 2 <= D < M")
    kwargs = {"M": m}
    if is_dpf:
        kwargs["D"] = intrinsic_d
    mop = CLASSES[name](**kwargs)
    start = time.perf_counter()
    ps = np.asarray(mop.ps(n_points=budget), dtype=np.float64)
    pf = np.asarray(mop.pf(n_points=budget), dtype=np.float64)
    seconds = time.perf_counter() - start
    replay = np.asarray(mop.evaluation(ps)["F"], dtype=np.float64)
    duplicate_run = np.asarray(mop.pf(n_points=budget), dtype=np.float64)

    if ps.ndim != 2 or ps.shape[1] != mop.N:
        raise AssertionError(f"{name}: invalid PS shape")
    if pf.ndim != 2 or pf.shape[1] != m:
        raise AssertionError(f"{name}: invalid PF shape")
    if replay.shape != (len(ps), m):
        raise AssertionError(f"{name}: evaluation shape mismatch")
    if not all(np.isfinite(x).all() for x in (ps, pf, replay)):
        raise AssertionError(f"{name}: nonfinite GT")
    if not np.array_equal(pf, duplicate_run):
        raise AssertionError(f"{name}: nondeterministic PF")
    if not np.all((ps >= 0) & (ps <= 1)):
        raise AssertionError(f"{name}: decision outside [0,1]")

    u = np.unique(pf, axis=0)
    rank = NonDominatedSorting().do(u, only_non_dominated_front=True)
    match = float(np.max(np.abs(pf - replay))) if len(pf) == len(replay) else None
    report = {
        "name": name,
        "M": m,
        "D": intrinsic_d if is_dpf else None,
        "N": mop.N,
        "requested": budget,
        "PS_rows": int(len(ps)),
        "PF_rows": int(len(pf)),
        "unique_objective_rows": int(len(u)),
        "objective_duplicates_exact_float64": int(len(pf) - len(u)),
        "unique_objectives_internally_dominated": int(len(u) - len(rank)),
        "deterministic": True,
        "source": _source_kind(name, m),
        "PS_PF_ordered_replay_max_abs": match,
        "observed_generator_seconds": float(seconds),
        "spacing": _spacing(u),
        "global_Pareto_membership_proven_by_this_audit": False,
        "continuous_PF_coverage_certified_by_this_audit": False,
    }

    if name in ("DTLZ1", "DPF1"):
        base = replay if name == "DTLZ1" else replay[:, :intrinsic_d]
        residual = np.abs(np.sum(base, axis=1) - 0.5)
        report["simplex_sum_half_max_abs"] = float(np.max(residual))
    if name in ("DTLZ2", "DTLZ3", "DTLZ4"):
        report["positive_unit_sphere_max_abs"] = float(
            np.max(np.abs(np.linalg.norm(pf, axis=1) - 1))
        )
        axes = np.eye(m)
        report["max_axis_to_reference_distance"] = float(
            np.max(cKDTree(u).query(axes, k=1)[0])
        )
    if name in ("DPF1", "DPF2", "DPF4"):
        if name == "DPF4":
            thresholds = mop._chaos_pool[:m-intrinsic_d]
            report["projection_thresholds_in_ascending_order"] = bool(
                np.all(np.diff(thresholds) >= 0)
            )
            report["published_ascending_thresholds_required"] = True
            published = _published_dpf4_at_ps(
                ps, m, intrinsic_d, thresholds
            )
            mismatch = np.abs(replay - published)
            report["published_sorted_DPF4_max_component_delta"] = float(
                np.max(mismatch)
            )
            report["published_sorted_DPF4_disagreeing_rows_tol_1e_12"] = int(
                np.sum(np.any(mismatch > 1e-12, axis=1))
            )
        if name == "DPF2":
            report["record_patches"] = _record_component_count(
                ps[:, :intrinsic_d-1]
            )
    if name == "DTLZ7":
        report["record_patches"] = _record_component_count(ps[:, :m-1])

    if name in ("DTLZ8", "DTLZ9"):
        result = mop.evaluation(ps, n_ieq_constr=mop.get_n_ieq_constr())
        g = np.asarray(result["G"])
        report["feasibility"] = {
            "strict_infeasible": int(np.sum(~result["feasible"])),
            "infeasible_above_1e-12": int(np.sum(np.any(g > 1e-12, axis=1))),
            "max_positive_residual": float(np.max(np.maximum(g, 0))),
        }
        if name == "DTLZ8":
            witness = dtlz8_same_t_pareto_witness(pf)
            improvement = pf - witness
            # A constructive witness with the SAME last objective must
            # satisfy every published DTLZ8 constraint. A strict
            # improvement above 1e-10 proves dominated GT rows.
            feasible = ~np.any(g > 1e-10, axis=1)
            dominance = (
                feasible & np.all(improvement >= -1e-10, axis=1) &
                np.any(improvement > 1e-10, axis=1)
            )
            # Validate witness independently from the incumbent PS.
            witness_g = mop._calc_constraints(witness)
            report["same_fM_independent_pareto_witness"] = {
                "strictly_dominated_returned_rows_tol_1e_10": int(np.sum(dominance)),
                "max_coordinate_improvement": float(np.max(np.maximum(improvement, 0))),
                "max_witness_constraint_violation": float(
                    np.max(np.maximum(-witness_g, 0))
                ),
                "global_membership_proven_if_not_dominated": False,
            }
    if name == "DTLZ9":
        square = pf[:, -1]**2 + pf[:, 0]**2
        report["circle_constraint_max_abs"] = float(
            np.max(np.abs(square - 1))
        )
        report["first_M_minus_1_objectives_equal_max_abs"] = float(
            np.max(np.abs(pf[:, :-1] - pf[:, :1]))
        )

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--names", nargs="+", choices=sorted(CLASSES),
                        default=sorted(CLASSES))
    parser.add_argument("--include-intrinsic-three", action="store_true")
    parser.add_argument("--probe-dtlz8-cap", action="store_true")
    args = parser.parse_args()
    records = []
    for name in args.names:
        for m in args.objectives:
            for budget in args.budgets:
                records.append(audit(name, m, budget))
                if args.include_intrinsic_three and name.startswith("DPF") and m > 3:
                    records.append(audit(name, m, budget, intrinsic_d=3))
    if args.probe_dtlz8_cap:
        for m in (3, 5, 10):
            records.append(audit("DTLZ8", m, 4096))

    print(json.dumps({
        "schema": "gt119_remaining_mop_matrix_v1",
        "scope": "necessary conditions; no global completeness certification",
        "excluded_deferred_DTLZ5_DTLZ6": "Issue #65",
        "records": records,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
