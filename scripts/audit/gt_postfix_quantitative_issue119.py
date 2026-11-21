# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only post-fix GT finite-sample evidence for MoeaBench issue #119.

This is an empirical diagnostic, not a certification of continuous
Hausdorff distance, canonical mixed-dimensional measure, or MOEA
performance. It never writes GT baselines, cache or certificates.
"""
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np
from scipy.spatial import cKDTree

from moeabench.mops.DTLZ1 import DTLZ1
from moeabench.mops.DTLZ2 import DTLZ2
from moeabench.mops.DTLZ3 import DTLZ3
from moeabench.mops.DTLZ4 import DTLZ4
from moeabench.mops.DTLZ7 import DTLZ7
from moeabench.mops.DTLZ8 import DTLZ8
from moeabench.mops.DTLZ9 import DTLZ9
from moeabench.mops.DPF1 import DPF1
from moeabench.mops.DPF2 import DPF2
from moeabench.mops.DPF3 import DPF3
from moeabench.mops.DPF4 import DPF4
from moeabench.mops.DPF5 import DPF5
from moeabench.mops._dtlz7_pareto_regions import dtlz7_record_intervals

PROBLEMS = {
    "DTLZ1": DTLZ1, "DTLZ2": DTLZ2, "DTLZ3": DTLZ3,
    "DTLZ4": DTLZ4, "DTLZ7": DTLZ7, "DTLZ8": DTLZ8,
    "DTLZ9": DTLZ9, "DPF1": DPF1, "DPF2": DPF2,
    "DPF3": DPF3, "DPF4": DPF4, "DPF5": DPF5,
}


def make_problem(name, m=3, d=2):
    if name not in PROBLEMS:
        raise ValueError(f"Unknown MOP: {name}")
    params = {"M": m}
    if name.startswith("DPF"):
        params["D"] = d
    return PROBLEMS[name](**params)


def region_counts(name, f, x, d):
    """Count record component occupancy in decision parameter space.

    The count does NOT imply geometric regularity of any component.
    """
    if name not in ("DTLZ7", "DPF2"):
        return None
    dimension = f.shape[1]-1 if name == "DTLZ7" else d-1
    ((_, high1), (low2, _)) = dtlz7_record_intervals()
    positions = x[:, :dimension]
    allowed = np.logical_or(positions <= high1 + 1e-12,
                            positions >= low2 - 1e-12)
    ids = np.sum((positions >= low2).astype(np.int64) *
                 (1 << np.arange(dimension, dtype=np.int64)), axis=1)
    return {
        "possible": 2**dimension,
        "occupied": len(np.unique(ids)),
        "invalid_record_rows": int(np.count_nonzero(~np.all(allowed, axis=1))),
    }


def geometry_residual(name, f):
    """Independently computable geometry only; null for other MOPs."""
    m = f.shape[1]
    if name == "DTLZ1":
        return float(np.max(np.abs(np.sum(f, axis=1) - 0.5)))
    if name in ("DTLZ2", "DTLZ3", "DTLZ4"):
        return float(np.max(np.abs(np.linalg.norm(f, axis=1) - 1)))
    if name == "DTLZ9":
        return float(np.max(np.abs(f[:, :-1]**2 + f[:, -1:]**2 - 1)))
    if name == "DTLZ8":
        t, y = f[:, -1], f[:, :-1]
        if m == 2:
            return float(np.max(np.abs(y[:, 0] - (1-t)/4)))
        line = t >= 1/3
        residuals = []
        if np.any(line):
            residuals.append(float(np.max(np.abs(
                y[line] - ((1-t[line])/4)[:, None]))))
        if np.any(~line):
            z = np.sort(y[~line], axis=1)
            residuals.append(float(np.max(np.abs(
                z[:, 0] + z[:, 1] - (1-2*t[~line])))))
            residuals.append(float(np.max(np.maximum(
                (1-t[~line])/4 - z[:, 0], 0))))
            if m > 3:
                residuals.append(float(np.max(np.abs(
                    z[:, 2:] - z[:, 1:2]))))
        return max(residuals) if residuals else 0.
    return None


def _spacing(f):
    unique = np.unique(f, axis=0)
    if len(unique) < 2:
        return {"nn_mean": None, "nn_cv": None, "nn_max": None}
    nearest = cKDTree(unique).query(unique, k=2, workers=1)[0][:, 1]
    mean = float(nearest.mean())
    return {
        "nn_mean": mean, "nn_cv": float(nearest.std()/mean) if mean else None,
        "nn_max": float(nearest.max()),
    }



def component_spacing(name, f, x, d):
    """Label individual sampled components; never pool their spacing."""
    m = f.shape[1]
    labels = None
    if name in ("DTLZ7", "DPF2"):
        dim = m-1 if name == "DTLZ7" else d-1
        low2 = dtlz7_record_intervals()[1][0]
        labels = np.sum((x[:, :dim] >= low2).astype(np.int64) *
                        (1 << np.arange(dim, dtype=np.int64)), axis=1)
    elif name == "DTLZ8":
        t = f[:, -1]
        labels = np.zeros(len(f), dtype=np.int64)
        if m >= 3:
            patch = t < 1/3
            labels[patch] = 1 + np.argmin(f[patch, :-1], axis=1)
    if labels is None:
        return None
    return {str(int(k)): {"count": int(np.count_nonzero(labels == k)),
                           **_spacing(f[labels == k])}
            for k in np.unique(labels)}

def sample(name, budget, m=3, d=2):
    mop = make_problem(name, m=m, d=d)
    tic = time.perf_counter()
    x = np.asarray(mop.ps(n_points=budget), dtype=np.float64)
    f = np.asarray(mop.evaluation(x)["F"], dtype=np.float64)
    elapsed = time.perf_counter() - tic
    check = np.asarray(mop.ps(n_points=budget), dtype=np.float64)
    stable = bool(np.array_equal(x, check))
    if f.ndim != 2 or len(f) == 0 or not np.isfinite(f).all():
        raise RuntimeError(f"{name}: invalid objective array")
    if x.shape != (len(f), mop.N):
        raise RuntimeError(f"{name}: objective/decision mapping mismatch")
    result = {
        "problem": name, "M": m, "D": d if name.startswith("DPF") else None,
        "budget_requested": budget, "budget_actual": len(f),
        "unique_objectives": len(np.unique(f, axis=0)),
        "ps_reproducible": stable, "elapsed_seconds": elapsed,
        "objective_sha256": hashlib.sha256(f.tobytes()).hexdigest(),
        "geometry_residual": geometry_residual(name, f),
        "region_occupancy": region_counts(name, f, x, d),
        "overall_spacing": _spacing(f),
        "component_spacing": component_spacing(name, f, x, d),
    }
    # An overall NN CV is a diagnostic only and cannot be interpreted
    # as regularity across disconnected or mixed-dimensional components.
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mops", nargs="+", choices=sorted(PROBLEMS),
                   default=sorted(PROBLEMS))
    p.add_argument("--budgets", nargs="+", type=int, default=[128, 512])
    p.add_argument("--objectives", nargs="+", type=int, default=[3, 5])
    p.add_argument("--large-budget", action="store_true")
    args = p.parse_args()
    records = []
    for name in args.mops:
        for m in args.objectives:
            if name.startswith("DPF"):
                d = min(3, m-1) if m >= 3 else 2
                if m <= d:
                    continue
            else:
                d = 2
            budgets = list(args.budgets)
            if args.large_budget:
                budgets.append(10000 if name.startswith("DPF") else 2000)
            for budget in sorted(set(budgets)):
                record = sample(name, budget, m=m, d=d)
                records.append(record)
    print(json.dumps({
        "schema": "moeabench_issue119_postfix_finite_gt_v1",
        "source_revision": os.environ.get("GITHUB_SHA"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "records": records,
        "continuous_coverage_certified": False,
        "canonical_component_measure_assumed": False,
        "historical_baselines_modified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
