# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent objective-space Pareto oracles for issue #119.

Constructs geometry from published equations without calling the PS/PF
generators or their private geometry helpers. The objective evaluator
is separately checked against each oracle. Reports finite-probe
distances, NOT a continuous-front certificate or component weights.
"""
import argparse
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree

from scripts.audit.gt_postfix_quantitative_issue119 import make_problem

SUPPORTED = ("DTLZ8", "DPF3", "DPF4", "DPF5")


def _chaos(n):
    # Published logistic recurrence, computed separately from BaseDPF.
    x = 0.1
    result = []
    for _ in range(n):
        x = 3.8 * x * (1.0 - x)
        result.append(x)
    return np.sort(result)


def _sphere(angles):
    n, q = angles.shape
    cos, sin = np.cos(angles), np.sin(angles)
    out = np.empty((n, q + 1))
    for j in range(q + 1):
        out[:, j] = np.prod(cos[:, :q-j], axis=1)
        if j:
            out[:, j] *= sin[:, q-j]
    return out


def _project_dpf34(y, m, squared):
    n, d = y.shape
    eta = _chaos(m-d)
    redundant = [np.minimum(y[:, -1], eta[0])]
    redundant.extend(np.clip(y[:, -1], eta[i], eta[i+1])
                     for i in range(m-d-1))
    redundant.append(np.maximum(y[:, -1], eta[-1]))
    f = np.column_stack((y[:, :-1], *redundant))
    return f**2 if squared else f


def independent_probes(name, m, d, count, seed):
    if name not in SUPPORTED or m < 3 or not 2 <= d < m:
        raise ValueError("Unsupported geometry or dimension.")
    rng = np.random.default_rng(seed)
    mop = make_problem(name, m=m, d=d)

    if name == "DTLZ8":
        groups = {}
        t = rng.uniform(1/3, 1, count)
        line = np.empty((count, m))
        line[:, :-1] = ((1-t)/4)[:, None]
        line[:, -1] = t
        groups["line"] = line
        for k in range(m-1):
            t = rng.uniform(0, 1/3, count)
            u = rng.uniform(0, 1, count)
            low = (1-t)/4 + u*((1-2*t)/2-(1-t)/4)
            high = 1 - 2*t - low
            f = np.repeat(high[:, None], m, axis=1)
            f[:, k] = low
            f[:, -1] = t
            groups[f"patch_{k}"] = f
        return groups

    if name in ("DPF3", "DPF4"):
        unit = rng.uniform(0, 1, (count, d-1))
        theta = unit * (np.pi/2)
        # The inverted spherical essentials are 1 - spherical direction,
        # independent of the numerically stabilized production formula.
        essential = 1 - _sphere(theta)
        return {"front": _project_dpf34(
            essential, m, squared=(name == "DPF4"))}

    groups = {}
    for label in ("P", "D"):
        q = m-1 if label == "P" else d-1
        u = rng.uniform(0, 1, (count, q))
        if label == "P":
            u[:, 0] *= (1/3)
            theta = u * np.pi/2
            groups[label] = _sphere(theta)
        else:
            u[:, 0] = 1/3 + (2/3) * u[:, 0]
            essential = _sphere(u * np.pi/2)
            repeat = m-d+1
            groups[label] = np.column_stack(
                (np.repeat(essential[:, :1]/np.sqrt(repeat),
                           repeat, axis=1), essential[:, 1:]))
    return groups


def independent_decisions(name, m, d, count, seed):
    """Separate valid decision population for evaluator-oracle crosscheck."""
    if name == "DTLZ8":
        return None  # objective equations independently establish this PF
    rng = np.random.default_rng(seed)
    mop = make_problem(name, m=m, d=d)
    x = np.full((count, mop.N), 0.5)
    if name in ("DPF3", "DPF4"):
        angles = rng.uniform(0, 1, (count, d-1))
        x[:, :d-1] = angles**(1/100) if name == "DPF3" else angles
    else:
        n_p = count // 2
        x[:, :m-1] = rng.uniform(0, 1, (count, m-1))
        x[:n_p, 0] *= 1/3
        x[n_p:, 0] = 1/3 + x[n_p:, 0]*2/3
        x[:, m-1] = x[:, 0]
    return x


def evaluator_independent_residual(name, m, d, count=64, seed=11953):
    if name == "DTLZ8":
        return None
    x = independent_decisions(name, m, d, count, seed)
    expected = None
    if name in ("DPF3", "DPF4"):
        theta = ((x[:, :d-1]**100) if name == "DPF3"
                 else x[:, :d-1]) * np.pi/2
        expected = _project_dpf34(
            1-_sphere(theta), m, squared=(name == "DPF4"))
    else:
        choose_d = x[:, 0] >= 1/3
        p = _sphere(x[:, :m-1]*np.pi/2)
        essential = _sphere(x[:, :d-1]*np.pi/2)
        repeat = m-d+1
        df = np.column_stack((
            np.repeat(essential[:, :1]/np.sqrt(repeat), repeat, axis=1),
            essential[:, 1:]))
        expected = np.where(choose_d[:, None], df, p)
    observed = np.asarray(make_problem(name, m, d).evaluation(x)["F"])
    return float(np.max(np.abs(observed-expected)))


def audit(name, m=3, d=2, budget=128, count=128, seed=11941):
    mop = make_problem(name, m=m, d=d)
    sample = np.asarray(mop.pf(n_points=budget))
    groups = independent_probes(name, m, d, count, seed)
    if sample.shape != (budget, m):
        raise AssertionError("Incorrect production PF cardinality.")
    residual = evaluator_independent_residual(name, m, d)
    records = {}
    for label, probes in groups.items():
        gap = cKDTree(sample).query(probes, k=1, workers=1)[0]
        records[label] = {
            "probes": len(probes),
            "maximum_gap": float(gap.max()),
            "mean_gap": float(gap.mean()),
        }
    return {
        "problem": name, "M": m, "D": d if name.startswith("DPF") else None,
        "budget": budget, "probe_per_component": count, "seed": seed,
        "source_pf_hash": hashlib.sha256(sample.tobytes()).hexdigest(),
        "evaluator_residual": residual, "components": records,
        "continuous_covering_certified": False,
        "mixed_dimensional_weight_assumed": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objectives", nargs="+", type=int, default=[3, 5])
    p.add_argument("--budgets", nargs="+", type=int, default=[128, 512])
    p.add_argument("--seeds", nargs="+", type=int, default=[11941,11942,11943])
    p.add_argument("--probes", type=int, default=128)
    args = p.parse_args()
    rows = [audit(name, m=m, d=min(3,m-1), budget=n,
                  count=args.probes, seed=seed)
            for name in SUPPORTED for m in args.objectives
            for n in args.budgets for seed in args.seeds]
    print(json.dumps({
        "schema": "gt_issue119_independent_published_geometry_v1",
        "records": rows,
        "continuous_covering_certified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
