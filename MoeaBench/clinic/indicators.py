
"""
MoeaBench Clinical Indicators
=============================

This module implements "Clinical Metrics" - calibrated variants of standard
MOO metrics (IGD, EMD) that are normalized by statistical baselines.

Purpose:
--------
To provide scale-invariant efficiency ratios that account for:
1. Finite Population Budget (N)
2. Intrinsic Dimensionality (M)
3. Discretization Noise of Ground Truth

Key Metrics:
------------
- igd_efficiency(P, GT, problem): R = IGD_obs / IGD_floor(N)
- emd_efficiency(P, GT, problem): R = EMD_obs / EMD_floor(N)
- purity_robust(P, GT): GD at 95th percentile (outlier resistant)

Architecture:
-------------
Ideally, this module is consumed by `mb.diagnostics.auditor`, ensuring
the auditor makes decisions based on "Efficiency" rather than "Raw Distance".
"""

import os
import json
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, Dict

# Lazy Load Cache
_BASELINES: Optional[Dict] = None

def _load_baselines():
    global _BASELINES
    if _BASELINES is None:
        path = os.path.join(os.path.dirname(__file__), "../diagnostics/resources/baselines.json")
        try:
            with open(path, "r") as f:
                _BASELINES = json.load(f)
        except Exception as e:
            # Fallback for CI or fresh install
            # print(f"Warning: Clinical Baselines not found at {path}. Using Identity.")
            _BASELINES = {"problems": {}}
    return _BASELINES

def get_floor(problem: str, metric: str, n: int = 200, stat: str = "p10") -> float:
    """
    Retrieves the statistical floor for a given problem/metric configuration.
    
    Args:
        problem (str): "DTLZ1", "DPF3", etc.
        metric (str): "igd_floor" or "emd_floor".
        n (int): Population budget (currently only support for 200).
        stat (str): "p10" (Expert), "p50" (Typical), "p90" (Pessimistic).
    
    Returns:
        float: The floor value. Returns 1e-9 if not found to avoid div/0.
    """
    bases = _load_baselines()
    try:
        # Currently N is implicit in the file, assuming generated for N=200
        # Future: baselines[problem][n][metric][stat]
        return bases["problems"][problem][metric][stat]
    except KeyError:
        # Fallback: Return a very small number -> efficiency will be huge (FAIL)
        # OR return 0.0 if handled upstream.
        # Ideally we log a warning.
        return 1e-9

def igd_efficiency(raw_igd: float, problem: str, n: int = 200) -> float:
    """
    Calculates the IGD Efficiency Ratio.
    
    R = raw_igd / igd_floor_p10
    
    Interpret:
      1.0: Perfect (matches best random sampling)
      <1.0: Better than random sampling (Smart Search)
      >1.5: Inefficient
    """
    floor = get_floor(problem, "igd_floor", n, "p10")
    if floor <= 1e-8: return 999.0 # Safe fallback for missing data
    return raw_igd / floor

def emd_efficiency(raw_emd: float, problem: str, n: int = 200) -> float:
    """
    Calculates the EMD Efficiency Ratio (Topology Check).
    
    R = raw_emd / emd_floor_p10
    """
    floor = get_floor(problem, "emd_floor", n, "p10")
    if floor <= 1e-8: return 999.0
    return raw_emd / floor

def purity_robust(P: np.ndarray, GT: np.ndarray, percentile: int = 95) -> float:
    """
    Calculates the Robust Purity (GD) at a high percentile.
    
    Instead of Mean GD (sensitive to one outlier), we aggregate the distance
    of the 95th percentile worst point. This trims the 5% worst garbage.
    
    Args:
        P (np.ndarray): Population points.
        GT (np.ndarray): Ground Truth points.
        percentile (int): 95 recommended.
        
    Returns:
        float: GD_p95
    """
    # 1. Distances from each point in P to nearest in GT
    dists = cdist(P, GT, metric='euclidean')
    min_dists = np.min(dists, axis=1) # Shape (N,)
    
    # 2. Percentile
    return float(np.percentile(min_dists, percentile))
