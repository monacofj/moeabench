# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Clinical Baseline Generator (v2)
==========================================

Generates the STRICT offline baselines for the Clinical Pathology Matrix.
Saves to `MoeaBench/diagnostics/resources/baselines_v2.json`.

Coverage:
- All MOPs (DTLZ1-7, DPF1-?)
- Grid K: [10, ..., 49] + [50, 100, 150, 200]
- Metrics: FIT, COVERAGE, GAP, UNIFORMITY, BALANCE

Logic:
- UNI50: Median metric of 30 deterministic FPS subsets (seed varies? NO. Review: FPS is deterministic seed 0 for REF, but for baseline distribution we need variance? 
  WAIT. The Peer Review said:
  "uni50 = median of FPS subsets..." 
  FPS is usually deterministic given a start point. 
  "rand50 = median of Random subsets..."
  
  CLARIFICATION: If FPS is deterministic (seed 0), then uni50 is a single value, not a distribution median.
  However, often we vary the start point of FPS to get a distribution of "optimal samplings".
  Let's assume "FPS with random start point" for the distribution of Uni50.
  
  And Reference U_K is FPS(seed=0).
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure project root in path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import MoeaBench.clinic.indicators as clinic
import MoeaBench.clinic.baselines as base

GT_DIR = os.path.join(PROJ_ROOT, "tests/ground_truth")
OUTPUT_JSON = os.path.join(PROJ_ROOT, "MoeaBench/diagnostics/resources/baselines_v2.json")

def get_fps_subset(gt, k, seed):
    # Use the helper but allow seed variation for distribution
    # Just reusing base.get_ref_uk with different seeds
    # But base.get_ref_uk sets rng = RandomState(seed) inside.
    return base.get_ref_uk(gt, k, seed=seed)

def get_random_subset(gt, k, seed):
    rng = np.random.RandomState(seed)
    if len(gt) <= k: return gt
    idx = rng.choice(len(gt), k, replace=False)
    return gt[idx]

def get_random_bbox(gt, k, seed):
    rng = np.random.RandomState(seed)
    mini = np.min(gt, axis=0)
    maxi = np.max(gt, axis=0)
    return mini + rng.rand(k, gt.shape[1]) * (maxi - mini)

def compute_baselines_for_mop(mop_name, gt_path):
    print(f"  Processing {mop_name}...")
    try:
        F_opt = pd.read_csv(gt_path, header=None).values
    except:
        print(f"    Simple CSV read failed, trying standard load...")
        F_opt = pd.read_csv(gt_path).values[:, :3] # Robustness

    # Pre-compute static references (Seed 0) that Indicators depend on
    # BUT: Indicators use U_K(seed=0) internally passed as arg.
    
    # Grid of K
    # Full range 10..49 + standard points
    k_grid = list(range(10, 50)) + [50, 100, 150, 200]
    
    mop_data = {}
    
    # Cache factors
    s_gt = base.get_resolution_factor(F_opt)
    
    for k in k_grid:
        if k > len(F_opt):
            continue
            
        # 1. Generate References for this K (The "Target")
        U_ref = base.get_ref_uk(F_opt, k, seed=0)
        C_cents, _ = base.get_ref_clusters(F_opt, 32, seed=0)
        
        # Hist Ref
        d = base.cdist(U_ref, C_cents)
        lab = np.argmin(d, axis=1)
        H_ref = np.bincount(lab, minlength=len(C_cents)).astype(float)
        H_ref /= np.sum(H_ref)
        
        # 2. Collect Samples for Distribution (N=30 repeats)
        # We need "Uni" (Near-Optimal) distribution and "Rand" (Random) distribution
        
        # Metrics: fit, coverage, gap, uniformity, balance
        # Accumulators
        uni_vals = {m: [] for m in ["fit", "coverage", "gap", "uniformity", "balance"]}
        rand_vals = {m: [] for m in ["fit", "coverage", "gap", "uniformity", "balance"]}
        
        for i in range(30):
            # Seed 1..30 (avoid 0 which is reference)
            seed = i + 1
            
            # Subsets
            # Uni: FPS with random start
            P_uni = get_fps_subset(F_opt, k, seed=seed)
            # Rand: Random Subset
            P_rand = get_random_subset(F_opt, k, seed=seed)
            # Rand BBox (for FIT only)
            P_bbox = get_random_bbox(F_opt, k, seed=seed)
            
            # -- FIT --
            # Note: Indicators.fit_q normalizes internally. 
            # But calculating baseline VALUES, we need raw metric.
            # We implemented indicators to take RAW P and compute Q.
            # Here we need to compute the RAW baselines to SAVE them.
            
            # Helper: Raw Computations
            def raw_fit(P): 
                d = base.cdist(P, F_opt)
                return np.percentile(np.min(d, axis=1), 95) / s_gt
            
            def raw_cov(P):
                d = base.cdist(F_opt, P)
                return np.mean(np.min(d, axis=1))

            def raw_gap(P):
                d = base.cdist(F_opt, P)
                return np.percentile(np.min(d, axis=1), 95)
                
            def raw_uni(P):
                # W1 vs U_ref
                d_p = base.cdist(P, P); np.fill_diagonal(d_p, np.inf); nn_p = np.min(d_p, axis=1)
                d_u = base.cdist(U_ref, U_ref); np.fill_diagonal(d_u, np.inf); nn_u = np.min(d_u, axis=1)
                return clinic.wasserstein_distance(nn_p, nn_u)

            def raw_bal(P):
                # JS vs H_ref
                d_p = base.cdist(P, C_cents); lab_p = np.argmin(d_p, axis=1)
                hp = np.bincount(lab_p, minlength=len(C_cents)).astype(float); hp /= np.sum(hp)
                return clinic.jensenshannon(hp, H_ref, base=2.0)
            
            # Collect Uni (FPS) values
            # Fit for Uni is 0 (subset), but let's compute to validade
            uni_vals["fit"].append(0.0) # Theoretical
            uni_vals["coverage"].append(raw_cov(P_uni))
            uni_vals["gap"].append(raw_gap(P_uni))
            uni_vals["uniformity"].append(raw_uni(P_uni))
            uni_vals["balance"].append(raw_bal(P_uni))

            # Collect Rand values
            # Fit uses BBox
            rand_vals["fit"].append(raw_fit(P_bbox))
            # Others use Rand Subset
            rand_vals["coverage"].append(raw_cov(P_rand))
            rand_vals["gap"].append(raw_gap(P_rand))
            rand_vals["uniformity"].append(raw_uni(P_rand))
            rand_vals["balance"].append(raw_bal(P_rand))
            
        # 3. Aggregates (Median)
        k_entry = {}
        for m in ["fit", "coverage", "gap", "uniformity", "balance"]:
            k_entry[m] = {
                "uni50": float(np.median(uni_vals[m])),
                "rand50": float(np.median(rand_vals[m]))
            }
        
        mop_data[str(k)] = k_entry
        
    return mop_data

def main():
    print("Starting Offline Baseline Generation (v2)...")
    
    # 1. Discover GT files
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*_3_optimal.csv")))
    all_data = {"problems": {}}
    
    for gt_f in gt_files:
        fname = os.path.basename(gt_f)
        mop_name = fname.replace("_3_optimal.csv", "")
        
        data = compute_baselines_for_mop(mop_name, gt_f)
        all_data["problems"][mop_name] = data
        
    # 2. Save JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_data, f, indent=2)
        
    print(f"\nDone! Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
