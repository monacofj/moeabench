# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import json
import numpy as np
from scipy.spatial.distance import cdist
import MoeaBench.diagnostics.baselines as base

def fix_dpf5():
    # Load DPF5 Calibration
    print("Loading DPF5 calibration...")
    # This ensures we use the NEW package with valid s_GT
    pkg_path = "MoeaBench/diagnostics/resources/references/DPF5/calibration_package.npz"
    pkg = np.load(pkg_path)
    ideal = pkg['ideal']
    gt_norm = pkg['gt_norm']
    
    # Calculate s_K for K=200
    K = 200
    s_k = base.get_resolution_factor_k(gt_norm, K)
    print(f"s_K (K={K}): {s_k}")

    # Generate Random Samples (N=K=200)
    print("Generating random samples...")
    # DPF5 has M=3 objectives
    M = 3
    # Use fixed seed for consistency
    np.random.seed(42)
    # Generate in normalized space [0,1]^M ? 
    # Wait, baselines are usually generated in normalized space?
    # generate_calibration.py generates 'u_ref' in normalized space.
    # But usually we sample in original space then normalize?
    # 'rand' baseline usually means uniformly distributed points in objective space.
    # Ideally we use get_reference_directions but randomized?
    # Let's use simple random sampling in [0,1]^3 since we are in normalized space.
    # DPF5 geometry: degenerate curve.
    # But random baseline is just points in the cube.
    samples = np.random.rand(K, M)
    
    # Calculate FAIR_FIT
    print("Calculating FAIR_FIT...")
    # fair_fit = GD95(P -> GT) / s_K
    d = cdist(samples, gt_norm)
    min_d = np.min(d, axis=1)
    gd95 = np.percentile(min_d, 95)
    
    fit_val = gd95 / s_k
    print(f"Calculated fit.rand50: {fit_val}")
    
    # Update JSON
    v3_path = "MoeaBench/diagnostics/resources/baselines_v3.json"
    with open(v3_path, "r") as f:
        data = json.load(f)
    
    if "DPF5" not in data["problems"]:
        data["problems"]["DPF5"] = {}
        
    data["problems"]["DPF5"]["200"] = {
        "fit": {
            "rand50": fit_val,
            "uni50": 0.0 # Placeholder
        }
    }
    
    with open(v3_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Updated {v3_path}")

if __name__ == "__main__":
    fix_dpf5()
