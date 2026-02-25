
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Baseline Generator v4 (ECDF-based)
============================================

Generates baselines_v4.json containing ECDF profiles for ALL clinical metrics:
1. HEADWAY (Progress better than noise)
2. CLOSENESS (Proximity to GT via Normal-Blur)
3. COV (Coverage)
4. GAP (Continuity)
5. REG (Regularity)
6. BAL (Balance)

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
from typing import Any

# Ensure project root in path for imports
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import json
import numpy as np
import MoeaBench.diagnostics.baselines as base
import MoeaBench.diagnostics.fair as fair
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

# Configuration
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../../MoeaBench/diagnostics/resources/baselines_v0.12.0.json")
N_SAMPLES = 200  # Size of the ECDF (Physics of Failure)
N_IDEAL = 30     # Size of Ideal population for determining uni50
SEED_START = 1000

# Full K Grid for Fixed K-Policy
K_VALUES = list(range(10, 51)) + [100, 150, 200]

def estimate_gt_normals(gt: np.ndarray) -> np.ndarray:
    """
    Estimates unit normals for each GT point using local PCA with KDTree for speed.
    """
    from scipy.spatial import KDTree
    N, M = gt.shape
    
    # Use KDTree for fast neighborhood search
    tree = KDTree(gt)
    k_pca = int(np.clip(np.round(np.sqrt(N)), 10, 30))
    
    # Query all neighbors at once
    _, indices = tree.query(gt, k=k_pca + 1)
    
    normals = np.zeros((N, M))
    for i in range(N):
        # Neighbors (exclude self)
        neighbors = gt[indices[i, 1:]]
        
        # Local PCA
        mean = np.mean(neighbors, axis=0)
        centered = neighbors - mean
        cov = np.dot(centered.T, centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Normal is the eigenvector of the smallest eigenvalue
        n = eigvecs[:, 0]
        
        # Force outward orientation (pointing away from origin)
        if np.dot(n, gt[i]) < 0:
            n = -n
            
        normals[i] = n / np.linalg.norm(n)
        
    return normals

def generate_half_normal_blur(gt: np.ndarray, normals: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Generates a blurred version of GT using Half-Normal displacement along normals.
    Includes domain safety (coordinates >= 0).
    """
    N, M = gt.shape
    blurred = np.zeros_like(gt)
    
    for i in range(N):
        n = normals[i]
        valid = False
        retry = 0
        mag_scale = 1.0
        
        while not valid and retry < 5:
            # Half-Normal magnitude
            mag = abs(rng.normal(0, sigma * mag_scale))
            # Random sign
            sign = rng.choice([-1, 1])
            
            # If normal is zero (fallback), use random isotropic direction
            if np.linalg.norm(n) < 1e-12:
                n_rand = rng.normal(size=M)
                n_rand /= np.linalg.norm(n_rand)
                p_prime = gt[i] + sign * mag * n_rand
            else:
                p_prime = gt[i] + sign * mag * n
                
            # Domain safety: Objective space [0, inf)
            if np.all(p_prime >= -1e-10):
                valid = True
                blurred[i] = p_prime
            else:
                # Retry strategy: try other sign, then reduce magnitude
                if retry == 0:
                    p_prime_flip = gt[i] - sign * mag * n
                    if np.all(p_prime_flip >= -1e-10):
                        valid = True
                        blurred[i] = p_prime_flip
                else:
                    mag_scale *= 0.5
                retry += 1
        
        if not valid:
            # Last resort: just stay at GT
            blurred[i] = gt[i]
            
    return blurred

def calibrate_sigma(gt: np.ndarray, normals: np.ndarray, s_fit: float, rng_seed: int, tree: Any = None) -> float:
    """
    Finds sigma such that median(min_dist(G_blur, GT) / s_fit) approx 2.0.
    Uses KDTree for fast distance calculation.
    """
    target = 2.0
    sigma = 2.0 * s_fit
    
    # Use tree if provided, else build one
    from scipy.spatial import KDTree
    _tree = tree if tree is not None else KDTree(gt)
    
    for it in range(6):
        rng = np.random.RandomState(rng_seed + it)
        blurred = generate_half_normal_blur(gt, normals, sigma, rng)
        
        # Compute min distances using KDTree
        d, _ = _tree.query(blurred, k=1)
        u_blur = d / s_fit
        
        med = np.median(u_blur)
        if abs(med - target) / target < 0.05:
            break
            
        # Multiplicative scaling
        if med > 1e-12:
            sigma *= (target / med)
        else:
            sigma *= 2.0
            
    # Final check of p95
    p95 = np.percentile(u_blur, 95)
    return sigma, med, p95

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
        dict: {headway, cov, gap, reg, bal}
    """
    metrics = {}
    
    # 1. HEADWAY (Proximity)
    metrics['headway'] = fair.headway(samples, gt_norm, s_k=s_fit).value

    # 2. COV (Coverage)
    metrics['cov'] = fair.coverage(samples, gt_norm).value

    # 3. GAP (Continuity)
    metrics['gap'] = fair.gap(samples, gt_norm).value

    # 4. REG (Regularity)
    metrics['reg'] = fair.regularity(samples, u_ref).value

    # 5. BAL (Balance)
    metrics['bal'] = fair.balance(samples, c_cents, hist_ref).value

    return metrics

def generate_ecdf_for_problem(prob_name, gt_norm):
    problem_data = {}
    M = gt_norm.shape[1] # Objectives

    # 1. Estimate Normals and build distance tree (once per problem)
    print(f" estimating normals...", end="", flush=True)
    from scipy.spatial import KDTree
    tree = KDTree(gt_norm)
    normals = estimate_gt_normals(gt_norm)

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
        
        # --- GENERATION (CLOSENESS BASELINE) ---
        # Part B2: Calibrate sigma and generate G_blur
        sigma, med_blur, p95_blur = calibrate_sigma(gt_norm, normals, s_fit, SEED_START + k, tree=tree)
        
        # Final baseline for this K
        rng_blur = np.random.RandomState(SEED_START + k + 500)
        g_blur = generate_half_normal_blur(gt_norm, normals, sigma, rng_blur)
        
        # u_blur = d_j / s_fit (Using KDTree)
        d_blur, _ = tree.query(g_blur, k=1)
        u_blur = d_blur / s_fit
        
        # Sort and downsample to N_SAMPLES if needed (to keep JSON size uniform)
        u_blur_sorted = np.sort(u_blur)
        if len(u_blur_sorted) > N_SAMPLES:
            # Linear sampling of the ECDF
            indices = np.linspace(0, len(u_blur_sorted)-1, N_SAMPLES).astype(int)
            u_blur_ecdf = u_blur_sorted[indices]
        else:
            u_blur_ecdf = u_blur_sorted
            
        # --- GENERATION (RANDOM) ---
        # "Failure" Baseline: Random Sampling in [0,1]^M
        rand_vals = {"headway": [], "cov": [], "gap": [], "reg": [], "bal": []}
        
        rng = np.random.RandomState(SEED_START + k)
        for _ in range(N_SAMPLES):
            # Random Pop
            pop = rng.rand(k, M)
            m = calculate_metrics(pop, gt_norm, s_fit, u_ref, c_cents, hist_ref)
            for key in rand_vals: rand_vals[key].append(m[key])
            
        # --- GENERATION (IDEAL) ---
        # "Success" Baseline: Uniform Sampling from GT (u_ref itself, or perturbations?)
        uni_vals = {"headway": [], "cov": [], "gap": [], "reg": [], "bal": []}
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
        for metric in ["headway", "cov", "gap", "reg", "bal"]:
            # Random (ECDF)
            r_all = np.sort(rand_vals[metric])
            r_50 = float(np.median(r_all))
            r_ecdf = [float(x) for x in r_all]
            
            # Ideal (Scalar Anchor)
            u_50 = float(np.median(uni_vals[metric]))
            
            # Special case for HEADWAY: Ideal is 0.0 by definition of distance
            if metric == "headway": u_50 = 0.0
            
            k_data[metric] = {
                "uni50": u_50,
                "rand50": r_50,
                "rand_ecdf": r_ecdf
            }
            
        # Store CLOSENESS
        # Ideal is degenerate 0.0
        k_data["closeness"] = {
            "uni50": 0.0,
            "rand50": float(med_blur),
            "rand_ecdf": [float(x) for x in u_blur_ecdf],
            "sigma": float(sigma),
            "p95": float(p95_blur),
            "seed": SEED_START + k
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
