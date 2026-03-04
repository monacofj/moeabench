
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import json
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Relative to diagnostics/clinic -> diagnostics/resources
RESOURCES_DIR = os.path.abspath(os.path.join(BASE_DIR, "../resources"))
V2_PATH = os.path.join(RESOURCES_DIR, "baselines_v2.json")
V3_PATH = os.path.join(RESOURCES_DIR, "baselines_v3.json")
REF_DIR = os.path.join(RESOURCES_DIR, "references")

def med_nn(pts):
    """Calculates median Nearest-Neighbor distance using KDTree for efficiency."""
    if len(pts) < 2:
        return 1.0
    tree = KDTree(pts)
    dist, _ = tree.query(pts, k=2)
    return float(np.median(dist[:, 1]))

def main():
    print(f"Loading baselines from {V2_PATH}...")
    if not os.path.exists(V2_PATH):
        print(f"Error: {V2_PATH} not found.")
        return

    with open(V2_PATH, "r") as f:
        data = json.load(f)

    problems = data.get("problems", {})
    print(f"DEBUG keys in loop: {list(problems.keys())}")
    new_problems = {}

    for prob_name in sorted(problems.keys()):
        k_dict = problems[prob_name]
        print(f"Processing {prob_name}...")
        new_k_dict = {}
        
        # Load Calibration Package
        pkg_path = os.path.join(REF_DIR, prob_name, "calibration_package.npz")
        if not os.path.exists(pkg_path):
            print(f"  Warning: Missing package for {prob_name} at {pkg_path}. Skipping.")
            continue
            
        pkg = np.load(pkg_path)
        gt_norm = pkg["gt_norm"]
        
        # Calculate s_GT once per problem
        s_gt = med_nn(gt_norm)
        print(f"  > s_GT = {s_gt:.6e}")

        for k_str, metrics in k_dict.items():
            k = int(k_str)
            
            # Extract strict U_K for this K
            u_key = f"u_ref_{k}"
            if u_key not in pkg:
                # Some K values in JSON might not be in the NPZ if K_GRID differs
                # We'll skip if not found to remain strictly deterministic
                print(f"    ! K={k} missing in package. Skipping.")
                continue
            
            u_ref_k = pkg[u_key]
            
            # Calculate s_K
            s_k = med_nn(u_ref_k)
            scale_factor = s_gt / s_k
            print(f"    > K={k}: s_K={s_k:.6e}, Factor={scale_factor:.4f}")
            
            # Clone metrics
            new_metrics = metrics.copy()
            
            if "fit" in new_metrics:
                old_rand = new_metrics["fit"]["rand50"]
                new_rand = old_rand * scale_factor
                new_metrics["fit"]["rand50"] = new_rand
                # print(f"      -> Rescaled fit.rand50: {old_rand:.2f} -> {new_rand:.2f}")
            
            new_k_dict[k_str] = new_metrics
            
        new_problems[prob_name] = new_k_dict
    
    data["problems"] = new_problems
    data["version"] = "3.0"
    if "meta" not in data: data["meta"] = {}
    data["meta"]["description"] = "Baselines re-scaled to finite-regime s_K units."
    
    print(f"Saving v3 to {V3_PATH}...")
    with open(V3_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print("Migration Complete.")

if __name__ == "__main__":
    main()
