# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import json
import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

# Ensure Project Root is in path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from MoeaBench.mops import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, DTLZ8, DTLZ9
from MoeaBench.mops import DPF1, DPF2, DPF3, DPF4, DPF5


SCHEMA_VERSION = "calib-v1"
K_GRID = [50, 100, 150, 200, 300]
B_RAND = 100
B_UNI = 30
DATA_DIR = os.path.join(PROJ_ROOT, "MoeaBench/diagnostics/resources/references")

MOPS = [
    DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, DTLZ8, DTLZ9,
    DPF1, DPF2, DPF3, DPF4, DPF5,
]


def _normalize(gt: np.ndarray):
    ideal = np.min(gt, axis=0)
    nadir = np.max(gt, axis=0)
    denom = nadir - ideal
    denom[denom == 0] = 1.0
    gt_norm = (gt - ideal) / denom
    # Remove duplicates to ensure non-zero s_GT
    _, idx = np.unique(np.round(gt_norm, 8), axis=0, return_index=True)
    gt_norm = gt_norm[np.sort(idx)]
    return gt_norm, ideal, nadir


def _k_center_indices(pts: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(pts)
    if k >= n:
        return np.arange(n, dtype=int)
    first = int(rng.integers(0, n))
    chosen = [first]
    # Track min distance to chosen set incrementally.
    min_d = cdist(pts, pts[[first]]).reshape(-1)
    min_d[first] = 0.0
    while len(chosen) < k:
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        d_new = cdist(pts, pts[[nxt]]).reshape(-1)
        min_d = np.minimum(min_d, d_new)
        min_d[nxt] = 0.0
    return np.array(chosen, dtype=int)


def _nn_dists(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.array([0.0])
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


def _metrics_for_subset(gt_norm: np.ndarray, subset_norm: np.ndarray, u_ref_norm: np.ndarray, s: float):
    # Coverage: GT -> subset
    d_gt_to = np.min(cdist(gt_norm, subset_norm), axis=1)
    igd_mean = float(np.mean(d_gt_to))
    igd_95 = float(np.percentile(d_gt_to, 95))

    # Purity: subset -> GT, normalized by resolution
    d_p_to = np.min(cdist(subset_norm, gt_norm), axis=1)
    gd_95 = float(np.percentile(d_p_to, 95))
    pur = float(gd_95 / s) if s > 1e-12 else float("inf")

    # Shape: Wasserstein on NN distance distribution vs quasi-uniform reference
    nn_p = _nn_dists(subset_norm)
    nn_u = _nn_dists(u_ref_norm)
    shape = float(wasserstein_distance(nn_p, nn_u))

    return {"IGD_mean": igd_mean, "IGD_95": igd_95, "PUR": pur, "SHAPE": shape}


def generate_calibration(mop_cls):
    mop = mop_cls()
    mop_name = mop_cls.__name__
    print(f"Generating GT calibration for {mop_name}...")

    if hasattr(mop, "pf"):
        gt = mop.pf(n_points=10000)
    else:
        try:
            gt = mop.optimal_front(n_points=10000)
        except Exception:
            gt = mop.optimal_front()

    gt_norm, ideal, nadir = _normalize(gt)
    gt_hash = hashlib.sha256(np.ascontiguousarray(gt_norm).tobytes()).hexdigest()

    out_dir = os.path.join(DATA_DIR, mop_name)
    os.makedirs(out_dir, exist_ok=True)

    pkg = {"gt_norm": gt_norm, "ideal": ideal, "nadir": nadir, "gt_hash": gt_hash, "schema_version": SCHEMA_VERSION}

    calib = {
        "mop": mop_name,
        "M": int(gt.shape[1]),
        "schema_version": SCHEMA_VERSION,
        "gt_hash": gt_hash,
        "K_grid": K_GRID,
        "B_rand": B_RAND,
        "B_uni": B_UNI,
        "metrics": {},
    }

    for K in K_GRID:
        K_eff = min(K, len(gt_norm))
        rng_ref = np.random.default_rng(0)
        idx_ref = _k_center_indices(gt_norm, K_eff, rng_ref)
        u_ref = gt_norm[idx_ref]
        pkg[f"u_ref_{K}"] = u_ref
        s = float(np.median(_nn_dists(u_ref)))

        rand_vals = {k: [] for k in ["IGD_mean", "IGD_95", "PUR", "SHAPE"]}
        uni_vals = {k: [] for k in ["IGD_mean", "IGD_95", "PUR", "SHAPE"]}

        rng = np.random.default_rng(12345 + K_eff)
        for _ in range(B_RAND):
            idx = rng.choice(len(gt_norm), K_eff, replace=False)
            sub = gt_norm[idx]
            m = _metrics_for_subset(gt_norm, sub, u_ref, s)
            for kk in rand_vals:
                rand_vals[kk].append(m[kk])

        for j in range(B_UNI):
            rng_u = np.random.default_rng(9000 + 31 * K_eff + j)
            idx = _k_center_indices(gt_norm, K_eff, rng_u)
            sub = gt_norm[idx]
            m = _metrics_for_subset(gt_norm, sub, u_ref, s)
            for kk in uni_vals:
                uni_vals[kk].append(m[kk])

        def p50(xs):
            return float(np.median(np.array(xs, dtype=float)))

        calib["metrics"][str(K)] = {
            "s": s,
            "IGD_mean": {"uni_p50": p50(uni_vals["IGD_mean"]), "rand_p50": p50(rand_vals["IGD_mean"])},
            "IGD_95": {"uni_p50": p50(uni_vals["IGD_95"]), "rand_p50": p50(rand_vals["IGD_95"])},
            "PUR": {"uni_p50": p50(uni_vals["PUR"]), "rand_p50": p50(rand_vals["PUR"])},
            "SHAPE": {"uni_p50": p50(uni_vals["SHAPE"]), "rand_p50": p50(rand_vals["SHAPE"])},
        }

    np.savez(os.path.join(out_dir, "calibration_package.npz"), **pkg)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)

    print(f"  Done: {out_dir}")


if __name__ == "__main__":
    for mop_cls in MOPS:
        try:
            generate_calibration(mop_cls)
        except Exception as e:
            print(f"Error processing {mop_cls.__name__}: {e}")
