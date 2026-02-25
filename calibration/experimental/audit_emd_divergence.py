# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Audit Script for Topic B: IGD vs EMD Divergence
==============================================
Reproduces the phenomenon where IGD is low (excellent) but EMD is high (poor).
Tests the hypothesis that EMD is sensitive to cardinality mismatch.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

# Ensure local MoeaBench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import MoeaBench as mb
from MoeaBench.metrics.evaluator import normalize

def emd_classic(F_a, F_b):
    """
    Naive 1D Wasserstein averaged over dimensions.
    (This is often used as a proxy for EMD in simple benchmarks, 
     though true EMD requires solving the transport plan)
    """
    # 1. Sort both
    # If sizes differ, we can't do direct 1-to-1 subtraction.
    # We must use scipy's wasserstein_distance which handles different weights/sizes for 1D.
    
    dists = []
    for m in range(F_a.shape[1]):
        u = F_a[:, m]
        v = F_b[:, m]
        dists.append(wasserstein_distance(u, v))
        
    return np.mean(dists)

def resampling_bootstrap(F, target_n=1000):
    """Resamples front F to have exactly target_n points."""
    indices = np.random.choice(len(F), target_n, replace=True)
    return F[indices]

def main():
    print("--- EMD Divergence Audit ---")
    
    # Target Problem from Review: DPF3 (or DTLZ7)
    mop_name = "DPF3"
    print(f"Target Problem: {mop_name}")
    
    # 1. Load Ground Truth
    gt_file = os.path.join(PROJ_ROOT, f"calibration/data/ground_truth/{mop_name}_3_optimal.csv")
    if not os.path.exists(gt_file):
        print("GT not found.")
        return
    F_opt = pd.read_csv(gt_file, header=None).values
    
    # 2. Load a solution file (NSGA2 which had high EMD)
    data_dir = os.path.join(PROJ_ROOT, "calibration/data/calibration_data")
    # Finding a standard run
    sol_file = None
    for f in os.listdir(data_dir):
        if mop_name in f and "NSGA2_standard_run00.csv" in f:
            sol_file = os.path.join(data_dir, f)
            break
            
    if not sol_file:
        print("Solution file not found.")
        return
        
    F_sol = pd.read_csv(sol_file).values
    if F_sol.shape[1] > 3: F_sol = F_sol[:, :3]
    
    print(f"GT Size: {len(F_opt)}")
    print(f"Sol Size: {len(F_sol)}")
    
    # 3. Normalize Shared Space
    min_val, max_val = normalize([F_opt], [F_sol])
    F_opt_norm = (F_opt - min_val) / (max_val - min_val + 1e-9)
    F_sol_norm = (F_sol - min_val) / (max_val - min_val + 1e-9)
    
    # 4. Calculate Current EMD (Naive 1D AVG)
    emd_raw = emd_classic(F_opt_norm, F_sol_norm)
    print(f"\n[Baseline] Raw EMD (sizes {len(F_opt)} vs {len(F_sol)}): {emd_raw:.6f}")
    
    # 5. Calculate with Resampling (Equal Cardinality)
    # We fix N=2000 for both
    target_n = 2000
    F_opt_res = resampling_bootstrap(F_opt_norm, target_n)
    F_sol_res = resampling_bootstrap(F_sol_norm, target_n)
    
    emd_res = emd_classic(F_opt_res, F_sol_res)
    print(f"[Experiment] Resampled EMD (N={target_n}): {emd_res:.6f}")
    
    # 6. Check Sliced Wasserstein (Approximation)
    # Project onto random directions
    n_projections = 100
    avg_sw = 0
    dirs = np.random.randn(n_projections, F_opt_norm.shape[1])
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    
    for d in dirs:
        # Project
        proj_opt = F_opt_norm @ d
        proj_sol = F_sol_norm @ d
        avg_sw += wasserstein_distance(proj_opt, proj_sol)
        
    avg_sw /= n_projections
    print(f"[Alternative] Sliced Wasserstein (Est.): {avg_sw:.6f}")
    
    # 7. Debug: Bounds & Vis
    print("\n[Debug] Normalized Bounds:")
    print(f"  GT  Min: {np.min(F_opt_norm, axis=0)} Max: {np.max(F_opt_norm, axis=0)}")
    print(f"  Sol Min: {np.min(F_sol_norm, axis=0)} Max: {np.max(F_sol_norm, axis=0)}")
    
    # Generate tiny debug plot
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=F_opt_norm[:,0], y=F_opt_norm[:,1], z=F_opt_norm[:,2],
        mode='markers', marker=dict(size=2, color='gray', opacity=0.3),
        name='Ground Truth (Norm)'
    ))
    fig.add_trace(go.Scatter3d(
        x=F_sol_norm[:,0], y=F_sol_norm[:,1], z=F_sol_norm[:,2],
        mode='markers', marker=dict(size=4, color='red'),
        name='Solution (Norm)'
    ))
    fig.update_layout(title=f"Debug {mop_name} EMD Divergence", scene=dict(aspectmode='cube'))
    out_file = os.path.join(PROJ_ROOT, "calibration/experimental/audit_debug.html")
    fig.write_html(out_file)
    print(f"Debug plot saved to: {out_file}")

    # 8. Sanity Check: What is the IGD?
    from pymoo.indicators.igd import IGD
    igd_metric = IGD(F_opt_norm, zero_to_one=True)
    igd_val = igd_metric.do(F_sol_norm)
    print(f"\n[Sanity] IGD Value: {igd_val:.6f}")
    
    # 9. Control Experiment: Perfect Subsample
    # What if the solution was just a random subset of GT?
    indices = np.random.choice(len(F_opt_norm), len(F_sol_norm), replace=False)
    F_control = F_opt_norm[indices]
    
    # Calculate EMD for control
    # Metric logic:
    dists_ctrl = []
    for i in range(F_control.shape[1]):
         w = wasserstein_distance(F_opt_norm[:,i], F_control[:,i])
         dists_ctrl.append(w)
    val_ctrl = np.mean(dists_ctrl)
    print(f"[Control] EMD of Perfect Subsample: {val_ctrl:.6f}")

    if val_ctrl > 0.1:
        print("\nCONCLUSION: EMD is broken/sensitive to sample size even for uniform distributions.")
    elif emd_raw > val_ctrl * 5:
        print("\nCONCLUSION: Algorithm is heavily biased (clumped). EMD is correct.")
    else:
        print("\nCONCLUSION: Ambiguous.")

if __name__ == "__main__":
    main()
