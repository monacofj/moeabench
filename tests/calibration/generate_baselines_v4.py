
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Baseline Generator v4 (ECDF-based)
============================================

Generates baselines_v4.json containing ECDF profiles for ALL clinical metrics:
1. FIT (Proximity)
2. COV (Coverage)
3. GAP (Continuity)
4. REG (Regularity)
5. BAL (Balance)

For each metric M and population size K:
- rand_ecdf: Sorted list of 200 values from Random Sampling (The "Failure" Profile)
- uni50: Median value from Uniform Sampling (The "Ideal" Profile)
- rand50: Median value from Random Sampling

Key Features:
- Deterministic sampling (Fixed Seeds).
- Full K coverage: Generates baselines for K=10..50, 100, 150, 200.
"""

import os
import sys

# Ensure project root in path for imports
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import json
import numpy as np
import MoeaBench.diagnostics.baselines as base
import MoeaBench.diagnostics.fair as fair
from scipy.spatial.distance import cdist

# Configuration
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../../MoeaBench/diagnostics/resources/baselines_v4.json")
N_SAMPLES = 200  # Size of the ECDF (Physics of Failure)
N_IDEAL = 30     # Size of Ideal population for determining uni50
SEED_START = 1000

# Full K Grid for Fixed K-Policy
K_VALUES = list(range(10, 51)) + [100, 150, 200]

def calculate_metrics(samples, gt_norm, s_fit, u_ref, c_cents, hist_ref):
    """
    Calculates all 5 Clinical Metrics for a given sample population.
    
    Args:
        samples (np.ndarray): The population to evaluate (N x M)
        gt_norm (np.ndarray): Ground Truth (Normalized)
        s_fit (float): Resolution factor for FIT
        u_ref (np.ndarray): Uniform Reference for REG
        c_cents (np.ndarray): Cluster centers for BAL
        hist_ref (np.ndarray): Reference histogram for BAL
        
    Returns:
        dict: {fit, cov, gap, reg, bal}
    """
    metrics = {}
    
    # 1. FIT (Proximity)
    metrics['fit'] = fair.compute_fair_fit(samples, gt_norm, s_fit)

    # 2. COV (Coverage)
    metrics['cov'] = fair.compute_fair_coverage(samples, gt_norm)

    # 3. GAP (Continuity)
    metrics['gap'] = fair.compute_fair_gap(samples, gt_norm)

    # 4. REG (Regularity)
    metrics['reg'] = fair.compute_fair_regularity(samples, u_ref)

    # 5. BAL (Balance)
    metrics['bal'] = fair.compute_fair_balance(samples, c_cents, hist_ref)

    return metrics

def generate_ecdf_for_problem(prob_name, gt_norm):
    problem_data = {}
    M = gt_norm.shape[1] # Objectives

    # Global References (Resolution Indep)
    # Clusters for Balance (Fixed seed for consistency)
    c_cents, _ = base.get_ref_clusters(gt_norm, c=32, seed=0)
    
    # Pre-compute Uniform Reference for Balance (Ideal Histogram)
    # We need a dense U_ref to compute the ideal histogram once
    # Actually fair.compute_fair_balance needs hist_ref derived from F_OPT (GT)
    # But wait, baselines.py says: "hist_ref passed to fair_balance".
    # In audit_calibration.py: 
    #   U_ref = base.get_ref_uk(F_opt, K_target, seed=0)
    #   d_u = base.cdist(U_ref, C_cents)
    #   lab_u = np.argmin(d_u, axis=1)
    #   hist_ref ...
    # Ideally, Balance should be checked against Uniform distribution on GT.
    # Let's follow auditor logic: derive hist_ref from a Uniform sampling of GT (U_ref).
    
    for k in K_VALUES:
        k_str = str(k)
        
        # --- PREPARATION ---
        # 1. s_K (Scale for FIT)
        s_fit = base.get_resolution_factor_k(gt_norm, k, seed=0)
        
        # 2. U_ref (Reference for REG and BAL) - The "Ideal" shape
        u_ref = base.get_ref_uk(gt_norm, k, seed=0)
        
        # 3. Reference Histogram for BAL
        # Map U_ref to clusters to define "Identity" distribution
        d_u = cdist(u_ref, c_cents)
        lab_u = np.argmin(d_u, axis=1)
        hist_ref = np.bincount(lab_u, minlength=len(c_cents)).astype(float)
        hist_ref /= np.sum(hist_ref)
        
        # --- GENERATION (RANDOM) ---
        # "Failure" Baseline: Random Sampling in [0,1]^M
        rand_vals = {"fit": [], "cov": [], "gap": [], "reg": [], "bal": []}
        
        rng = np.random.RandomState(SEED_START + k)
        for _ in range(N_SAMPLES):
            # Random Pop
            pop = rng.rand(k, M)
            m = calculate_metrics(pop, gt_norm, s_fit, u_ref, c_cents, hist_ref)
            for key in rand_vals: rand_vals[key].append(m[key])
            
        # --- GENERATION (IDEAL) ---
        # "Success" Baseline: Uniform Sampling from GT (u_ref itself, or perturbations?)
        # Ideally, we sample subsets of GT that are perfectly Uniform.
        # But u_ref IS the definition of Uniform on GT for size K.
        # So "Ideal" is effectively computing metrics on u_ref itself (or similar uniform sets).
        # Since u_ref is deterministic (seed=0), let's generate a few variations 
        # using different seeds for get_ref_uk to simulate "Ideal but stochastic" algorithms?
        # Specification says: "Ideal = Uni50 (FPS of GT)". 
        # So we sample N_IDEAL sets using FPS with different seeds.
        
        uni_vals = {"fit": [], "cov": [], "gap": [], "reg": [], "bal": []}
        for i in range(N_IDEAL):
            # Sample Uniformly from GT (Simulates a perfect algorithm)
            # Seed varies to get distribution of "Perfect"
            pop_uni = base.get_ref_uk(gt_norm, k, seed=100+i) 
            
            # Note: For REG, we compare pop_uni against u_ref (seed=0).
            # This captures the variance of "perfect" samplers against the canonical reference.
            m = calculate_metrics(pop_uni, gt_norm, s_fit, u_ref, c_cents, hist_ref)
            for key in uni_vals: uni_vals[key].append(m[key])

        # --- STORAGE ---
        k_data = {}
        for metric in ["fit", "cov", "gap", "reg", "bal"]:
            # Random (ECDF)
            r_all = np.sort(rand_vals[metric])
            r_50 = float(np.median(r_all))
            r_ecdf = [float(x) for x in r_all]
            
            # Ideal (Scalar Anchor)
            u_50 = float(np.median(uni_vals[metric]))
            
            # Special case for FIT: Ideal is 0.0 by definition of distance
            if metric == "fit": u_50 = 0.0
            
            k_data[metric] = {
                "uni50": u_50,
                "rand50": r_50,
                "rand_ecdf": r_ecdf
            }
            
        problem_data[k_str] = k_data
        
    return problem_data

def main():
    print(f"Generating Baselines v4 (ECDF)...")
    print(f"Output: {OUTPUT_FILE}")
    print(f"K_VALUES: {K_VALUES}")
    
    # Locate references
    # Assuming code is running from tests/calibration/
    ref_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../MoeaBench/diagnostics/resources/references"))
    
    problems = []
    if os.path.exists(ref_dir):
        for d in sorted(os.listdir(ref_dir)):
            if os.path.isdir(os.path.join(ref_dir, d)) and os.path.exists(os.path.join(ref_dir, d, "calibration_package.npz")):
                problems.append(d)
    
    if not problems:
        print(f"Error: No reference problems found in {ref_dir}. CWD: {os.getcwd()}")
        return

    print(f"Found {len(problems)} problems: {problems}")
    
    final_json = {
        "schema": "baselines_v4_ecdf",
        "rand_ecdf_n": N_SAMPLES,
        "problems": {}
    }
    
    for prob in problems:
        print(f"Processing {prob}...", end="", flush=True)
        try:
            pkg_path = os.path.join(ref_dir, prob, "calibration_package.npz")
            pkg = np.load(pkg_path)
            gt_norm = pkg['gt_norm']
            
            p_data = generate_ecdf_for_problem(prob, gt_norm)
            final_json["problems"][prob] = p_data
            print(" Done.")
            
        except Exception as e:
            print(f" Failed! ({e})")
            import traceback
            traceback.print_exc()

    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_json, f, indent=None) 
    
    print("Baseline generation complete.")

if __name__ == "__main__":
    main()
