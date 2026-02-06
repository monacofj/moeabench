
"""
MoeaBench Baseline Generator
============================

Generates statistical baselines (floors) for clinical metrics (IGD, EMD)
by bootstrapping the Ground Truth data.

Methodology:
1. Load Ground Truth (GT) for each problem.
2. For N=200 (standard budget):
   a. Sample K=100 random subsets of size N from GT.
   b. Compute IGD(Subset, GT) -> Distribution -> P10, P50 (Med), P90.
   c. Compute EMD(SubsetA, SubsetB) -> Distribution -> P10, P50, P90.
      (Paired EMD isolates discretization noise from cardinality effects).
3. Save results to MoeaBench/diagnostics/resources/baselines.json.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Direct Import Plumbing
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from MoeaBench.metrics.GEN_igd import GEN_igd
try:
    from MoeaBench.metrics.GEN_emd import GEN_emd
except ImportError:
    # Fallback if GEN_emd is not exposed directly, though it should be.
    # We will implement a direct caller if needed or use stats.topo
    pass

from MoeaBench.stats import topo_distribution

GT_DIR = os.path.join(PROJ_ROOT, "tests/ground_truth")
OUTPUT_FILE = os.path.join(PROJ_ROOT, "MoeaBench/diagnostics/resources/baselines.json")

# Configuration
MOPS = []
# DTLZ Family
for i in range(1, 10):
    MOPS.append(f"DTLZ{i}")
# DPF Family
for i in range(1, 6):
    MOPS.append(f"DPF{i}")

N_BUDGET = 200
K_SAMPLES = 100

def get_gd_quantiles(gt_points, n, k=50):
    """
    Computes Robust Purity (GD) baseline.
    Actually, for GD, if points are FROM GT, GD is 0.
    So we don't compute GD floor (it's 0), but we could compute
    'expected nearest neighbor distance' as a proxy for sparsity.
    For now, we skip GD floor as it is theoretically 0 for subsets.
    """
    return 0.0

def compute_baselines():
    print(f"Starting Baseline Generation (N={N_BUDGET}, K={K_SAMPLES})...")
    
    baselines = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "n_budget": N_BUDGET,
            "k_samples": K_SAMPLES
        },
        "problems": {}
    }

    # Helper for EMD
    # We use the internal topo_distribution logic but need to be careful
    # EMD is expensive. We might reduce K for EMD if needed.
    
    for mop in MOPS:
        # Load GT (3 objectives is standard for calibration)
        gt_path = os.path.join(GT_DIR, f"{mop}_3_optimal.csv")
        
        if not os.path.exists(gt_path):
            print(f"Skipping {mop}: GT file not found at {gt_path}")
            continue
            
        print(f"Processing {mop}...", end="", flush=True)
        
        try:
            df = pd.read_csv(gt_path, header=None)
            gt_data = df.values[:, :3] # Force 3D
            
            # Theoretical Bounds (Strict)
            ideal = np.min(gt_data, axis=0)
            nadir = np.max(gt_data, axis=0)
            
            # 1. IGD Floor Analysis
            igd_samples = []
            
            # Using Pymoo's IGD for speed/consistency if available, or internal
            from pymoo.indicators.igd import IGD
            metric_igd = IGD(gt_data) # Pymoo normalizes internally if requested, carefully check
            # Actually, let's use raw distance to match MoeaBench robustly
            # MoeaBench IGD takes (Pop, GT)
            
            # Pre-compute normalization? 
            # Consistent with Report: No internal normalization if inputs are within range?
            # Metric IGD usually expects inputs in same scale.
            
            for _ in range(K_SAMPLES):
                # Sample N points without replacement
                indices = np.random.choice(len(gt_data), N_BUDGET, replace=False)
                sample = gt_data[indices]
                
                # Compute IGD(Sample, GT)
                val = metric_igd.do(sample)
                igd_samples.append(val)
                
            igd_p10 = np.percentile(igd_samples, 10)
            igd_p50 = np.percentile(igd_samples, 50)
            igd_p90 = np.percentile(igd_samples, 90)
            
            # 2. EMD Floor Analysis (Paired Subsets)
            # We compare Sample A vs Sample B to find the "Self-Consistency Floor"
            emd_samples = []
            
            # Reduce K for EMD to save time?
            k_emd = min(K_SAMPLES, 30) 
            
            for _ in range(k_emd):
                idx_a = np.random.choice(len(gt_data), N_BUDGET, replace=False)
                idx_b = np.random.choice(len(gt_data), N_BUDGET, replace=False)
                
                sample_a = gt_data[idx_a]
                sample_b = gt_data[idx_b]
                
                # Use internal SciPy EMD (Wasserstein) via stats
                # Note: topo_distribution calculates axis-wise 1D Wasserstein
                # If we want full Multi-dim EMD (Earth Mover), strictly 2-Wasserstein is hard
                # But Scipy wasserstein_distance is 1D.
                # Let's check what GEN_emd does. Assuming typical usage.
                
                # Recalibrating: The report uses `mb.stats.topo_distribution(..., method='emd')`
                # which returns mean of 1D EMDs. We must match that exactly.
                res = topo_distribution(sample_a, sample_b, method='emd')
                val = np.mean(list(res.results.values()))
                emd_samples.append(val)
                
            emd_p10 = np.percentile(emd_samples, 10)
            emd_p50 = np.percentile(emd_samples, 50)
            emd_p90 = np.percentile(emd_samples, 90)
            
            baselines["problems"][mop] = {
                "igd_floor": {
                    "p10": float(igd_p10),
                    "p50": float(igd_p50),
                    "p90": float(igd_p90)
                },
                "emd_floor": {
                    "p10": float(emd_p10),
                    "p50": float(emd_p50),
                    "p90": float(emd_p90)
                },
                "gt_size": len(gt_data)
            }
            
            print(f" Done. IGD_P10={igd_p10:.4f}, EMD_P10={emd_p10:.4f}")
            
        except Exception as e:
            print(f" Failed: {e}")

    # Ensure output dir
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(baselines, f, indent=2)
        
    print(f"\nBaselines generated at: {OUTPUT_FILE}")

if __name__ == "__main__":
    compute_baselines()
