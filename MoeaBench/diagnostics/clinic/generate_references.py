# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import json
import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import logging

# Ensure Project Root is in path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from MoeaBench.mops import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, DTLZ8, DTLZ9
from MoeaBench.mops import DPF1, DPF2, DPF3, DPF4, DPF5

# Configuration
K_GRID = [50, 100, 200, 400, 800]
N_BOOTSTRAP = 200 # Sufficient for floors
DATA_DIR = os.path.join(PROJ_ROOT, "MoeaBench/diagnostics/resources/references")

MOPS = [
    DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, DTLZ8, DTLZ9,
    DPF1, DPF2, DPF3, DPF4, DPF5
]

def calculate_emd_1d_avg(pts, ref_pts):
    """Average 1D Wasserstein distance across objectives."""
    if pts is None or ref_pts is None: return np.nan
    M = pts.shape[1]
    dists = []
    for m in range(M):
        dists.append(wasserstein_distance(pts[:, m], ref_pts[:, m]))
    return np.mean(dists)

def get_unique_medoids(gt, K):
    """
    Selects K unique points from gt that best represent the distribution.
    """
    if len(gt) <= K:
        return gt, np.arange(len(gt))
    
    # 1. K-Means centroids
    # kmeans might return fewer than K centroids if points are too close
    centroids, _ = kmeans(gt, K, iter=20)
    K_found = len(centroids)
    
    # 2. Snap to nearest (Unique)
    dists = cdist(centroids, gt)
    medoid_indices = []
    used_indices = set()
    
    for i in range(K_found):
        sorted_indices = np.argsort(dists[i])
        for idx in sorted_indices:
            if idx not in used_indices:
                medoid_indices.append(idx)
                used_indices.add(idx)
                break
    
    # 3. Fill remaining if kmeans returned fewer centroids
    if len(medoid_indices) < K:
        # Just pick remaining points with largest distance to existing medoids
        while len(medoid_indices) < K:
            current_medoids = gt[medoid_indices]
            all_dists = cdist(gt, current_medoids)
            min_dists = np.min(all_dists, axis=1)
            # Find point farthest from any current medoid
            new_idx = np.argmax(min_dists)
            if new_idx in used_indices: # Should not happen with argmax
                 # Fallback: find first unused
                 for j in range(len(gt)):
                     if j not in used_indices:
                         new_idx = j
                         break
            medoid_indices.append(new_idx)
            used_indices.add(new_idx)
            
    return gt[medoid_indices], np.array(medoid_indices)

def generate_reference_package(mop_cls):
    mop_name = mop_cls.__name__
    print(f"Generating Reference Package for {mop_name}...")
    
    # 1. Instantiate MOP and get GT
    mop = mop_cls()
    if hasattr(mop, 'pf'):
        gt = mop.pf(n_points=10000)
    elif hasattr(mop, 'optimal_front'):
        # Some classes might not support n_points in optimal_front
        try: gt = mop.optimal_front(n_points=10000)
        except: gt = mop.optimal_front()
    else:
        print(f"  Warning: {mop_name} has no PF. Skipping.")
        return

    # 2. Normalization
    ideal = np.min(gt, axis=0)
    nadir = np.max(gt, axis=0)
    denom = nadir - ideal
    denom[denom == 0] = 1.0
    gt_norm = (gt - ideal) / denom
    
    out_dir = os.path.join(DATA_DIR, mop_name)
    os.makedirs(out_dir, exist_ok=True)
    
    pkg_data = {
        "gt_norm": gt_norm,
        "ideal": ideal,
        "nadir": nadir
    }
    
    baselines = {
        "mop": mop_name,
        "M": int(gt.shape[1]),
        "K_data": {}
    }

    for K in K_GRID:
        # Cap K by the size of the GT
        K_actual = min(K, len(gt_norm))
        print(f"  Processing K={K} (Actual K={K_actual})...")
        
        # 3. Uniform Reference (Medoids)
        ref_uni, medoid_indices = get_unique_medoids(gt_norm, K_actual)
        pkg_data[f"uni_{K}"] = ref_uni
        pkg_data[f"medoids_{K}"] = medoid_indices
        
        # 4. Bootstrap Floors
        igd_samples = []
        emd_uni_samples = []
        
        # Determine sampling size for bootstrap
        # If population is small, we can't do replace=False with K if K is too large.
        # But wait, the BOOTSTRAP is meant to simulate a population of size K.
        # If GT size < K, we should probably allow replacement or just cap it at len(gt).
        # Actually, if we are evaluating a population of size K, and the GT has fewer points,
        # the "ideal" is perfect overlap + duplicates.
        
        S_SIZE = K_actual
        
        for _ in range(N_BOOTSTRAP):
            # Random sub-sample of GT
            idx = np.random.choice(len(gt_norm), S_SIZE, replace=(S_SIZE > len(gt_norm)))
            sample = gt_norm[idx]
            
            # IGD Floor: mean(min_dist(GT_pts, sample))
            dists_igd = cdist(gt_norm, sample)
            igd_val = np.mean(np.min(dists_igd, axis=1))
            igd_samples.append(igd_val)
            
            # EMD Uniform Floor
            emd_val = calculate_emd_1d_avg(sample, ref_uni)
            emd_uni_samples.append(emd_val)
            
        baselines["K_data"][str(K)] = {
            "igd_floor": float(np.median(igd_samples)),
            "igd_p10": float(np.percentile(igd_samples, 10)),
            "emd_uni_floor": float(np.median(emd_uni_samples)),
            "emd_uni_p10": float(np.percentile(emd_uni_samples, 10))
        }

    # 5. Persist
    np.savez(os.path.join(out_dir, "ref_package.npz"), **pkg_data)
    with open(os.path.join(out_dir, "baselines.json"), "w") as f:
        json.dump(baselines, f, indent=4)
    
    print(f"  Done. Package saved to {out_dir}")

if __name__ == "__main__":
    for mop in MOPS:
        try:
            generate_reference_package(mop)
        except Exception as e:
            print(f"Error processing {mop.__name__}: {e}")
