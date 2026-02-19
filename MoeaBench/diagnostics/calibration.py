# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Calibration Engine (Automated MOP Profiling)
=====================================================

This module provides the logic to calculate Clinical Baselines (ECDF, Uni50, Rand50)
for any problem instance.
"""

import os
import json
import inspect
import numpy as np
from typing import Optional, List, Dict, Any, Union
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from . import baselines as base
from . import fair

# Defaults aligned with baselines_v4
N_SAMPLES_ECDF = 200
N_IDEAL_DRAWS = 30
K_GRID_DEFAULT = list(range(10, 51)) + [100, 150, 200]
SEED_START = 1000

def calibrate_mop(mop: Any, baseline: Optional[str] = None, force: bool = False, k_values: Optional[List[int]] = None) -> bool:
    """
    Programmatic entry point for MOP calibration.
    
    Logic:
    1. Resolve sidecar path (default: <mop_name>.json in MOP's directory).
    2. If exists and not force: load and register.
    3. Else: Perform full statistical profiling and save.
    
    Returns:
        bool: True if recalibrated, False if loaded from existing file.
    """
    mop_name = getattr(mop, 'name', mop.__class__.__name__)
    if not mop_name:
        mop_name = "UnknownMOP"

    # 1. Resolve Path
    if baseline:
        path = baseline
    else:
        # Sidecar resolution (proximity to class file)
        try:
            origin_file = inspect.getfile(mop.__class__)
            origin_dir = os.path.dirname(os.path.abspath(origin_file))
            path = os.path.join(origin_dir, f"{mop_name}.json")
        except:
            path = f"{mop_name}.json"

    # 2. Check Cache/File
    if os.path.exists(path) and not force:
        print(f"MoeaBench: Loading custom baselines for '{mop_name}' from sidecar.")
        base.register_baselines(path)
        return False

    # 3. Perform Calibration
    print(f"MoeaBench: Calibrating '{mop_name}' (this may take a minute)...")
    
    # Materialize Ground Truth
    gt = mop.pf(n_points=2000)
    if gt is None or len(gt) == 0:
        raise ValueError(f"MOP '{mop_name}' returned empty Pareto Front (pf). Cannot calibrate.")

    k_grid = k_values or K_GRID_DEFAULT
    problem_data = _generate_baselines(mop_name, gt, k_grid)

    # 4. Save Sidecar
    sidecar_data = {
        "problem_id": mop_name,
        "gt_reference": gt.tolist(),
        "problems": {
            mop_name: problem_data
        }
    }

    with open(path, "w") as f:
        json.dump(sidecar_data, f, indent=None)

    # 5. Register in current session
    base.register_baselines(sidecar_data)
    
    print(f"MoeaBench: Calibration complete. Profile saved to: {path}")
    return True

def _generate_baselines(name: str, gt: np.ndarray, k_grid: List[int]) -> Dict[str, Any]:
    """Internal math engine for baseline calculation."""
    M = gt.shape[1]
    
    # 1. Normals and Clusters
    normals = _estimate_gt_normals(gt)
    c_cents, _ = base.get_ref_clusters(gt, c=32, seed=0)
    
    problem_data = {}
    for k in k_grid:
        k_str = str(k)
        
        # Scale and references
        s_fit = base.get_resolution_factor_k(gt, k, seed=0)
        u_ref = base.get_ref_uk(gt, k, seed=0)
        
        # Balance hist
        d_u = cdist(u_ref, c_cents)
        lab_u = np.argmin(d_u, axis=1)
        hist_ref = np.bincount(lab_u, minlength=len(c_cents)).astype(float)
        hist_ref /= np.sum(hist_ref)
        
        # CLOSENESS (Blur)
        sigma, u_blur = _calibrate_and_blur(gt, normals, s_fit, SEED_START + k)
        u_blur_ecdf = _downsample_ecdf(np.sort(u_blur))
        
        # Metrics storage
        k_data = {
            "closeness": {
                "uni50": 0.0,
                "rand50": float(np.median(u_blur)),
                "rand_ecdf": [float(x) for x in u_blur_ecdf]
            }
        }
        
        # Other Metrics (Random and Ideal)
        rand_metrics = {m: [] for m in ["headway", "cov", "gap", "reg", "bal"]}
        ideal_metrics = {m: [] for m in ["headway", "cov", "gap", "reg", "bal"]}
        
        rng = np.random.RandomState(SEED_START + k)
        for _ in range(N_SAMPLES_ECDF):
            # Random Sample (Failure)
            pop_rand = rng.rand(k, M)
            m = _calc_all(pop_rand, gt, s_fit, u_ref, c_cents, hist_ref)
            for key in rand_metrics: rand_metrics[key].append(m[key])
            
        for i in range(N_IDEAL_DRAWS):
            # Ideal Sample (Success)
            pop_uni = base.get_ref_uk(gt, k, seed=100 + i)
            m = _calc_all(pop_uni, gt, s_fit, u_ref, c_cents, hist_ref)
            for key in ideal_metrics: ideal_metrics[key].append(m[key])
            
        # Consolidate
        for m in ["headway", "cov", "gap", "reg", "bal"]:
            r_all = np.sort(rand_metrics[m])
            k_data[m] = {
                "uni50": 0.0 if m == "headway" else float(np.median(ideal_metrics[m])),
                "rand50": float(np.median(r_all)),
                "rand_ecdf": [float(x) for x in r_all]
            }
            
        problem_data[k_str] = k_data
        
    return problem_data

def _calc_all(p, gt, s_fit, u_ref, c_cents, hist_ref):
    """Calculates all fair metrics for a sample."""
    return {
        "headway": fair.headway(p, gt, s_fit).value,
        "cov": fair.coverage(p, gt).value,
        "gap": fair.gap(p, gt).value,
        "reg": fair.regularity(p, u_ref).value,
        "bal": fair.balance(p, c_cents, hist_ref).value
    }

def _downsample_ecdf(sorted_vals: np.ndarray) -> np.ndarray:
    if len(sorted_vals) <= N_SAMPLES_ECDF:
        return sorted_vals
    indices = np.linspace(0, len(sorted_vals)-1, N_SAMPLES_ECDF).astype(int)
    return sorted_vals[indices]

def _calibrate_and_blur(gt, normals, s_fit, seed):
    """Calibrates sigma for Half-Normal blur."""
    target = 2.0
    sigma = 2.0 * s_fit
    
    # Iterative sigma search
    u_blur = None
    for it in range(6):
        rng = np.random.RandomState(seed + it)
        blurred = _generate_blur(gt, normals, sigma, rng)
        dists = cdist(blurred, gt)
        min_d = np.min(dists, axis=1)
        u_blur = min_d / s_fit
        med = np.median(u_blur)
        if abs(med - target) / target < 0.05:
            break
        sigma *= (target / med) if med > 1e-12 else 2.0
        
    return sigma, u_blur

def _generate_blur(gt, normals, sigma, rng):
    N, M = gt.shape
    blurred = np.zeros_like(gt)
    for i in range(N):
        n = normals[i]
        mag = abs(rng.normal(0, sigma))
        sign = rng.choice([-1, 1])
        if np.linalg.norm(n) < 1e-12:
            n_rand = rng.normal(size=M)
            n_rand /= np.linalg.norm(n_rand)
            p_prime = gt[i] + sign * mag * n_rand
        else:
            p_prime = gt[i] + sign * mag * n
        # Boundary [0, inf)
        blurred[i] = np.maximum(p_prime, 0.0)
    return blurred

def _estimate_gt_normals(gt: np.ndarray) -> np.ndarray:
    N = len(gt)
    M = gt.shape[1]
    k_pca = int(np.clip(np.round(np.sqrt(N)), 10, 30))
    dists = cdist(gt, gt)
    indices = np.argsort(dists, axis=1)[:, 1:k_pca+1]
    
    rows = np.repeat(np.arange(N), k_pca)
    cols = indices.flatten()
    data = np.ones(N * k_pca)
    adj = csr_matrix((data, (rows, cols)), shape=(N, N))
    n_components, labels = connected_components(adj, directed=False)
    
    normals = np.zeros((N, M))
    for i in range(N):
        comp_mask = (labels == labels[i])
        comp_indices = np.where(comp_mask)[0]
        k_eff = min(k_pca, len(comp_indices) - 1)
        if k_eff < 2: continue
        
        d_comp = dists[i, comp_mask]
        idx_in_comp = np.argsort(d_comp)[1:k_eff+1]
        neighbors = gt[comp_indices[idx_in_comp]]
        
        mean = np.mean(neighbors, axis=0)
        centered = neighbors - mean
        cov = np.dot(centered.T, centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        n = eigvecs[:, 0]
        normals[i] = n / (np.linalg.norm(n) + 1e-12)
        
    return normals
