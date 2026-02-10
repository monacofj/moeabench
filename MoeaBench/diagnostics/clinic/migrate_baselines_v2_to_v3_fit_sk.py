
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import json
import numpy as np
from scipy.spatial.distance import cdist

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Relative to diagnostics/clinic -> diagnostics/resources
RESOURCES_DIR = os.path.join(BASE_DIR, "../resources")
V2_PATH = os.path.join(RESOURCES_DIR, "baselines_v2.json")
V3_PATH = os.path.join(RESOURCES_DIR, "baselines_v3.json")
REF_DIR = os.path.join(RESOURCES_DIR, "references")

def calculate_sk(u_ref_k: np.ndarray) -> float:
    """Calculates s_K = median(NN(U_K))"""
    if len(u_ref_k) < 2:
        return 1.0
    d = cdist(u_ref_k, u_ref_k)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d, axis=1)
    return float(np.median(min_d))

def calculate_sgt(gt: np.ndarray) -> float:
    """Calculates s_GT = median(NN(GT))"""
    if len(gt) < 2:
        return 1.0
    d = cdist(gt, gt)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d, axis=1)
    return float(np.median(min_d))

def main():
    print(f"Loading baselines from {V2_PATH}...")
    with open(V2_PATH, "r") as f:
        data = json.load(f)

    problems = data.get("problems", {})
    new_problems = {}

    for prob_name, k_dict in problems.items():
        print(f"Processing {prob_name}...")
        new_k_dict = {}
        
        # Load Calibration Package
        pkg_path = os.path.join(REF_DIR, prob_name, "calibration_package.npz")
        if not os.path.exists(pkg_path):
            raise FileNotFoundError(f"Missing package for {prob_name} at {pkg_path}")
            
        pkg = np.load(pkg_path)
        gt_norm = pkg["gt_norm"]
        
        # Calculate s_GT once per problem
        s_gt = calculate_sgt(gt_norm)
        print(f"  > s_GT = {s_gt:.6e}")

        for k_str, metrics in k_dict.items():
            k = int(k_str)
            
            # Extract strict U_K for this K
            u_key = f"u_ref_{k}"
            if u_key not in pkg:
                 raise KeyError(f"Missing {u_key} in {pkg_path} for K={k}")
            
            u_ref_k = pkg[u_key]
            
            # Calculate s_K
            s_k = calculate_sk(u_ref_k)
            scale_factor = s_gt / s_k
            print(f"  > K={k}: s_K={s_k:.6e}, Factor={scale_factor:.4f}")
            
            # Clone metrics
            new_metrics = metrics.copy()
            
            if "fit" in new_metrics:
                old_rand = new_metrics["fit"]["rand50"]
                new_rand = old_rand * scale_factor
                new_metrics["fit"]["rand50"] = new_rand
                print(f"    -> Rescaled fit.rand50: {old_rand:.2f} -> {new_rand:.2f}")
            
            new_k_dict[k_str] = new_metrics
            
        new_problems[prob_name] = new_k_dict
    
    data["problems"] = new_problems
    data["version"] = "3.0"
    data["meta"]["description"] = "Baselines re-scaled to finite-regime s_K units."
    
    print(f"Saving v3 to {V3_PATH}...")
    with open(V3_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print("Migration Complete.")

if __name__ == "__main__":
    main()
