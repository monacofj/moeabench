# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Produce comparable FAIR-independent MoeaBench observations for two commits.

This script is audit instrumentation, NOT a replacement for existing tests.
"""
import argparse
import hashlib
import json
import warnings

import numpy as np

import moeabench as mb


def front_record(front):
    arr = np.asarray(front, dtype=np.float64)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise AssertionError("Invalid finite front")
    canonical = np.round(arr, 12).astype("<f8", copy=False)
    return {
        "shape": list(arr.shape),
        "digest_12dp": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "first": canonical[: min(3, len(canonical))].tolist(),
        "sum": np.sum(canonical, axis=0).tolist(),
    }


def matrix(result):
    out = np.asarray(result.values, dtype=float)
    if not np.isfinite(out).all():
        raise AssertionError("Nonfinite metric result")
    return out.tolist()


def observe():
    warnings.filterwarnings("ignore", message=".*Version Mismatch.*")
    p = np.array([[0.15, 0.85, 0.45], [0.4, 0.35, 0.7],
                  [0.9, 0.15, 0.2], [0.6, 0.55, 0.4]])
    q = np.array([[0.10, 0.80, 0.40], [0.3, 0.4, 0.6],
                  [0.75, 0.12, 0.25], [0.5, 0.6, 0.3]])
    reference = [p, q]
    out = {
        "schema": 1,
        "api": {
            "metrics": [hasattr(mb.metrics, x) for x in
                        ("hv", "gd", "gdplus", "igd", "igdplus", "emd", "front_ratio")],
            "stats": [hasattr(mb.stats, x) for x in
                      ("perf_compare", "topo_compare", "ranks", "strata", "tiers")],
        },
        "mops": {},
        "metrics": {},
    }

    for name, params in (
        ("DTLZ2", {"M": 3, "K": 5}),
        ("DTLZ7", {"M": 3}),
        ("DPF2", {"M": 3, "D": 2}),
        ("DPF4", {"M": 3, "D": 2}),
    ):
        mop = getattr(mb.mops, name)(**params)
        out["mops"][name] = front_record(mop.pf(n_points=320))

    for scale in ("raw", "rel"):
        out["metrics"][f"hv_{scale}_self"] = matrix(
            mb.metrics.hv([p, q], scale=scale, mode="exact", progress=False))
        out["metrics"][f"hv_{scale}_fixed"] = matrix(
            mb.metrics.hv(p, ref=reference, scale=scale, mode="exact", progress=False))
    for metric_name in ("gd", "gdplus", "igd", "igdplus", "emd"):
        fn = getattr(mb.metrics, metric_name)
        out["metrics"][metric_name] = matrix(fn(p, ref=q))

    mop = mb.mops.DTLZ2(M=3, K=5)
    alg = mb.moeas.NSGA2(population=24, generations=6, seed=123)
    exp = mb.experiment(mop=mop, moea=alg)
    exp.run(repeat=2, silent=True)
    out["experiment"] = {
        "seeds": exp.seed,
        "final": front_record(np.asarray(exp.last_run.front())),
        "hv_raw": matrix(mb.metrics.hv(exp, scale="raw", mode="exact", progress=False)),
        "hv_rel": matrix(mb.metrics.hv(exp, scale="rel", mode="exact", progress=False)),
    }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(observe(), f, sort_keys=True, indent=2)
