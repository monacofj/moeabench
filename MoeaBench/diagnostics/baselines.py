# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Clinical Baselines (Offline & Deterministic)
======================================================

This module handles:
1. Loading offline-computed baselines (Fail-Closed logic).
2. Generating deterministic reference objects ($U_K$, Clusters).
"""

import os
import json
import numpy as np
from typing import Optional, Dict, Any, Tuple
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2

# Path to the Authorized Offline Baselines
# Path to the Authorized Offline Baselines
BASELINE_JSON_PATH = os.path.join(os.path.dirname(__file__), "resources/baselines_v4.json")
_CACHE = None

class UndefinedBaselineError(Exception):
    """Raised when a required baseline is missing (Fail-Closed)."""
    pass

def load_offline_baselines() -> Dict[str, Any]:
    """
    Loads the strict offline baselines JSON.
    Raises UndefinedBaselineError if file missing.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    
    if not os.path.exists(BASELINE_JSON_PATH):
        # FAIL-CLOSED architecture: Raise error immediately
        raise UndefinedBaselineError(f"Baseline file not found at {BASELINE_JSON_PATH}")

    try:
        with open(BASELINE_JSON_PATH, "r") as f:
            _CACHE = json.load(f)
    except Exception as e:
        raise UndefinedBaselineError(f"Failed to parse baseline file: {e}")
    
    return _CACHE

def get_baseline_values(problem: str, k: int, metric: str) -> Tuple[float, float]:
    """
    Retrieves (uni50, rand50) for a given (problem, k, metric).
    
    Returns:
        (uni50, rand50)
        
    Raises:
        UndefinedBaselineError: If baseline is missing (Strict).
    """
    try:
        bases = load_offline_baselines()
    except UndefinedBaselineError:
        raise

    # Structure: problems -> {problem} -> {k} -> {metric} -> {uni50, rand50}
    try:
        p_data = bases.get("problems", {}).get(problem, {})
        k_data = p_data.get(str(k), {})
        m_data = k_data.get(metric, {})
        
        uni = m_data.get("uni50")
        rand = m_data.get("rand50")
        
        if uni is None or rand is None:
            raise ValueError("Values are None")
            
        return float(uni), float(rand)
        
    except (AttributeError, ValueError):
        raise UndefinedBaselineError(f"Missing baseline for {problem}, K={k}, {metric}")

def get_baseline_ecdf(problem: str, k: int, metric: str) -> Tuple[float, float, np.ndarray]:
    """
    Retrieves (uni50, rand50, rand_ecdf) for a given (problem, k, metric).
    
    Returns:
        (uni50, rand50, rand_ecdf)
        
    Raises:
        UndefinedBaselineError: If baseline is missing or invalid (Strict).
    """
    try:
        bases = load_offline_baselines()
    except UndefinedBaselineError:
        raise

    try:
        p_data = bases.get("problems", {}).get(problem, {})
        k_data = p_data.get(str(k), {})
        m_data = k_data.get(metric, {})
        
        uni = m_data.get("uni50")
        rand = m_data.get("rand50")
        rand_ecdf = m_data.get("rand_ecdf")
        
        if uni is None or rand is None or rand_ecdf is None:
            raise UndefinedBaselineError(f"Incomplete baseline for {problem}, K={k}, {metric}")
            
        rand_ecdf = np.array(rand_ecdf, dtype=float)
        
        # Strict Validation
        if len(rand_ecdf) != 200:
            raise UndefinedBaselineError(f"Invalid ECDF length {len(rand_ecdf)} for {problem}, K={k} (Expected 200)")
            
        # Check sorted (non-decreasing)
        if np.any(np.diff(rand_ecdf) < -1e-12): # Allow epsilon noise
             if np.any(np.diff(rand_ecdf) < 0):
                raise UndefinedBaselineError(f"ECDF not sorted for {problem}, K={k}")
        
        # Check Median Consistency (Optional but good)
        calc_median = np.median(rand_ecdf)
        if not np.isclose(rand, calc_median, atol=0.2): # Loosened for downsampled ECDFs
             # Log warning or fail? Fail-Closed means fail.
             # But floating point issues / downsampling might occur. tolerance 0.2 is safe for 200/1000 samples.
             raise UndefinedBaselineError(f"Baseline mismatch: rand50 ({rand}) != median(ecdf) ({calc_median}) for {problem}, K={k}")
            
        return float(uni), float(rand), rand_ecdf
        
    except (AttributeError, ValueError) as e:
        raise UndefinedBaselineError(f"Invalid baseline data for {problem}, K={k}: {e}")


def get_ref_uk(gt: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    Generates the Deterministic Reference $U_K$ (Uniform-at-K).
    
    Logic: Farthest Point Sampling (FPS) subset of GT.
    """
    if len(gt) <= k:
        return gt
        
    # Standard FPS (Greedy)
    # Using simple euclidean metric
    rng = np.random.RandomState(seed)
    # Start with random point to avoid bias of index 0 if GT is sorted
    start_idx = rng.randint(0, len(gt))
    
    indices = [start_idx]
    # Distances to current set
    min_dists = cdist(gt[start_idx:start_idx+1], gt, metric='euclidean')[0]
    
    for _ in range(k - 1):
        # Select point with max min_dist
        next_idx = np.argmax(min_dists)
        indices.append(next_idx)
        
        # Update distances
        new_dists = cdist(gt[next_idx:next_idx+1], gt, metric='euclidean')[0]
        min_dists = np.minimum(min_dists, new_dists)
        
    return gt[indices]

def get_ref_clusters(gt: np.ndarray, c: int = 32, seed: int = 0) -> Any:
    """
    Generates the Deterministic Reference Clusters (Centroids).
    
    Logic: K-Means on GT with fixed seed.
    Returns: (centroids, labels)
    """
    # Normalize GT to [0,1] for clustering stability if not already? 
    # Assuming GT is pre-normalized or typically [0,1] box.
    # Using scipy.cluster.vq.kmeans2
    n_pts = len(gt)
    if n_pts < c:
        # Fallback if GT is tiny: just use GT points as centroids
        return gt, np.arange(n_pts)
        
    centroids, labels = kmeans2(gt, c, minit='points', seed=seed, iter=20)
    return centroids, labels

def get_resolution_factor(gt: np.ndarray) -> float:
    """
    Calculates s_GT: Median Nearest Neighbor distance within GT.
    Used for normalizing HEADWAY/FIT.
    """
    if len(gt) < 2:
        return 1.0 # Fallback
        
    d = cdist(gt, gt)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d, axis=1)
    return float(np.median(min_d))

def get_resolution_factor_k(gt: np.ndarray, k: int, seed: int = 0) -> float:
    """
    Calculates s_K: Median Nearest Neighbor distance within U_K (FPS of GT).
    Used for normalizing HEADWAY/FIT in finite sampling regimes.
    """
    u_ref = get_ref_uk(gt, k, seed=seed)
    if len(u_ref) < 2:
        return 1.0
        
    d = cdist(u_ref, u_ref)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d, axis=1)
    return float(np.median(min_d))

def snap_k(k_raw: int) -> int:
    """
    Snaps the raw population size K to the nearest supported baseline K
    using a strict 'floor' logic for consistency with the audit protocol.
    
    Rules:
    - K < 10: Returns 10 (Minimum supported)
    - 10 <= K < 50: Returns K (Exact match)
    - K >= 50: Returns max(grid <= K) where grid=[50, 100, 150, 200]
    """
    if k_raw < 10:
        return 10
    if k_raw < 50:
        return k_raw
    
    grid = [50, 100, 150, 200]
    # Filter grid items <= k_raw
    candidates = [g for g in grid if g <= k_raw]
    if not candidates:
        # Should not happen given k_raw >= 50 check, but safe fallback
        return 50
    return max(candidates)
