# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
moeabench Clinical FR Metrics (Physical Layer)
================================================

This module implements the "FR" / "Clinical" metrics logic.
These metrics are:
1.  Physically grounded (Distances, Divergences).
2.  Corrected for basic scale artifacts (e.g., divided by resolution).
3.  Direction: Lower is Better (0.0 = Perfect Physical Match).
4.  NOT normalized to [0,1] quality scores (see qscore.py for that).
"""

import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import wasserstein_distance
from typing import Optional, Tuple, Any
from .utils import _resolve_diagnostic_context
from .base import DiagnosticValue

# Internal cache for KDtrees to avoid reconstruction in Monte Carlo loops
_TREE_CACHE = {}

def _get_kdtree(data):
    from scipy.spatial import KDTree
    # Use (shape, id) as key to avoid dimension leaks and handle array reallocations.
    key = (data.shape, id(data))
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = KDTree(data)
    return _TREE_CACHE[key]

def clear_fair_cache():
    global _TREE_CACHE
    _TREE_CACHE = {}
    from .baselines import clear_baselines_cache
    clear_baselines_cache()

class FairResult(DiagnosticValue):
    """ Specialized result for Physical (FR) metrics. """
    def report(self, show: bool = True, **kwargs) -> str:
        if kwargs.get('markdown', True):
            content = f"**{self.name}** (Physical): {self.value:.4f}\n- *Meaning*: {self.description}"
        else:
            content = f"{self.name} (Physical): {self.value:.4f}\n  Meaning: {self.description}"
        return self._render_report(content, show, **kwargs)

# Compatibility Alias
FairResult = FairResult


def headway(data: Any, ref: Optional[Any] = None, s_k: Optional[float] = None, **kwargs) -> FairResult:
    r"""
    [Smart API] Calculates HEADWAY (Longitudinal Search Drive).
    
    Definition: $GD_{95}(P_{final} \to GT) / GD_{95}(P_{initial} \to GT)$
    Meaning: The fraction of the initial search error that remains unreduced.
    Ideal: 0.0 (Search successfully reached the target).
    Failure: 1.0 (Algorithm did not move from its starting position).
    
    If 'data' is an Experiment/Run, P_initial is automatically extracted from Gen 0.
    Otherwise, users can provide 'initial_data' via kwargs.
    """
    ctx = _resolve_diagnostic_context(data, ref, s_k, **kwargs)
    P_f = ctx['P_final']
    P_i = ctx['P_initial']
    GT = ctx['GT']
    res = ctx['s_k']
    
    if P_f is None or GT is None or len(P_f) == 0:
        return FairResult(np.inf, "HEADWAY", "No points to evaluate.", raw_data=np.array([]))

    # Helper to compute GD95 (Distance to GT)
    def _get_gd95(P):
        if P is None or len(P) == 0: return np.nan
        if len(GT) > 1000 or True: # Use tree consistently for safety
            tree = _get_kdtree(GT)
            d, _ = tree.query(P, k=1)
            min_d = d
        else:
            d = cdist(P, GT, metric='euclidean')
            min_d = np.min(d, axis=1)
        return float(np.percentile(min_d, 95)), min_d

    # 1. Compute Final Distance
    dist_f, min_d_f = _get_gd95(P_f)
    u_vals_f = min_d_f / res if res > 1e-12 else min_d_f
    
    if kwargs.get('raw', False):
        return FairResult(
            value=dist_f / res if res > 1e-12 else dist_f,
            name="HEADWAY",
            description=f"Final distance is {dist_f/res if res > 1e-12 else dist_f:.2f} resolution units.",
            raw_data=u_vals_f
        )

    # 2. Compute Initial Distance for the Progress Ratio
    if P_i is None:
        # Fallback: if we don't have P_0, we cannot calculate search drive fairly.
        return FairResult(
            value=np.nan,
            name="HEADWAY",
            description="Unavailable: Longitudinal context (Pop 0) missing. Pass Experiment/Run object.",
            raw_data=u_vals_f
        )
    
    dist_i, _ = _get_gd95(P_i)
    
    # 3. Calculate Ratio (Residual Search Error)
    # If dist_i is 0 (unlikely in real search), headway is 1.0 unless dist_f is also 0.
    f_val = dist_f / dist_i if dist_i > 1e-12 else (1.0 if dist_f > 1e-12 else 0.0)
    
    return FairResult(
        value=float(f_val),
        name="HEADWAY",
        description=f"Algorithm left {f_val*100:.1f}% of the initial search error unreduced.",
        raw_data=u_vals_f
    )

def closeness(data: Any, ref: Optional[Any] = None, s_k: Optional[float] = None, **kwargs) -> FairResult:
    """
    [Smart API] Calculates the distribution of normalized distances to GT.
    
    Returns u_j = min_dist(p_j, GT) / s_fit
    """
    ctx = _resolve_diagnostic_context(data, ref, s_k, **kwargs)
    P = ctx['P_final']
    GT = ctx['GT']
    s_fit = ctx['s_k']
    problem_name = ctx['problem']
    k = ctx['k']

    if P is None or GT is None or len(P) == 0:
        return FairResult(0.0, "CLOSENESS", "No points to evaluate.", raw_data=np.array([]))
        
    # 1. Compute Distances to GT (Optimized via KDTree)
    tree = _get_kdtree(GT)
    d, _ = tree.query(P, k=1)
    min_d = d
    
    # 2. Normalize by Resolution
    u_vals_raw = min_d / s_fit if s_fit > 1e-12 else min_d
    
    # 3. Apply Plausible Ideal Correction (Residue Subtraction)
    from . import baselines as base
    ideal_res = 0.0
    try:
        if problem_name and k:
            b_data = base.get_baseline_data(problem_name, k, "closeness")
            ideal_res = b_data.get("ideal_res", 0.0)
    except:
        pass
    
    u_vals = np.maximum(0.0, u_vals_raw - ideal_res)
    f_val = float(np.median(u_vals))
    
    return FairResult(
        value=f_val,
        name="CLOSENESS",
        description=f"Median distance to ground truth manifold (corrected by ideal residue: {ideal_res:.4f}).",
        raw_data=u_vals
    )

def coverage(data: Any, ref: Optional[Any] = None, **kwargs) -> FairResult:
    r"""
    [Smart API] Calculates COVERAGE.
    
    Definition: $IGD_{mean}(GT \to P)$
    Meaning: Average distance from ANY point in GT to the nearest point in P.
    Ideal: 0.0 (Complete coverage)
    """
    ctx = _resolve_diagnostic_context(data, ref, **kwargs)
    P = ctx['P_final']
    GT = ctx['GT']

    if P is None or GT is None or len(P) == 0:
        return FairResult(np.inf, "COVERAGE", "No points to evaluate.", raw_data=np.array([]))

    # 1. Compute Distances from GT to P (Optimized via KDTree)
    tree = _get_kdtree(P)
    d, _ = tree.query(GT, k=1)
    min_d = d
    
    # 2. Mean (IGD)
    f_val = float(np.mean(min_d))
    return FairResult(
        value=f_val,
        name="COVERAGE",
        description=f"Average distance from target manifold to nearest solution is {f_val:.4f}.",
        raw_data=min_d
    )

def gap(data: Any, ref: Optional[Any] = None, **kwargs) -> FairResult:
    r"""
    [Smart API] Calculates GAP (formerly Density).
    
    Definition: $IGD_{95}(GT \to P)$
    Meaning: The size of the "large holes" in coverage (ignoring worst 5% outliers).
    Ideal: 0.0
    """
    ctx = _resolve_diagnostic_context(data, ref, **kwargs)
    P = ctx['P_final']
    GT = ctx['GT']

    if P is None or GT is None or len(P) == 0:
        return FairResult(np.inf, "GAP", "No points to evaluate.", raw_data=np.array([]))

    # 1. Compute Distances from GT to P (Optimized via KDTree)
    tree = _get_kdtree(P)
    d, _ = tree.query(GT, k=1)
    min_d = d
    
    # 2. Percentile 95 (Robust Max)
    f_val = float(np.percentile(min_d, 95))
    return FairResult(
        value=f_val,
        name="GAP",
        description=f"Largest hole detected on the manifold is {f_val:.4f}.",
        raw_data=min_d
    )

def regularity(data: Any, ref_distribution: Optional[np.ndarray] = None, **kwargs) -> FairResult:
    r"""
    [Smart API] Calculates REGULARITY (formerly Uniformity).
    
    Definition: $W_1(d_{NN}(P), d_{NN}(U_{ref}))$
    Meaning: Wasserstein distance between the Nearest-Neighbor distribution of P
             and that of a reference distribution (usually from a Uniform lattice).
    Ideal: 0.0
    """
    ctx = _resolve_diagnostic_context(data, **kwargs)
    P = ctx['P_final']
    U_ref = ref_distribution

    if P is None or len(P) < 2 or U_ref is None or len(U_ref) < 2:
        return FairResult(np.inf, "REGULARITY", "Insufficient points for regularity.", raw_data=np.array([]))

    # 1. Nearest Neighbors within P (excluding self)
    d_p = cdist(P, P)
    np.fill_diagonal(d_p, np.inf)
    nn_p = np.min(d_p, axis=1)
    
    # 2. Nearest Neighbors within Reference U_ref
    d_u = cdist(U_ref, U_ref)
    np.fill_diagonal(d_u, np.inf)
    nn_u = np.min(d_u, axis=1)
    
    # 3. Wasserstein Distance
    f_val = float(wasserstein_distance(nn_p, nn_u))
    return FairResult(
        value=f_val,
        name="REGULARITY",
        description=f"Deviation from ideal lattice spacing is {f_val:.4f} (Wasserstein distance).",
        raw_data=nn_p # Plot the distribution of NN distances
    )

def balance(data: Any, centroids: Optional[np.ndarray] = None, ref_hist: Optional[np.ndarray] = None, **kwargs) -> FairResult:
    r"""
    [Smart API] Calculates BALANCE.
    
    Definition: $D_{JS}(H_P || H_{ref})$
    Meaning: Jensen-Shannon divergence between cluster occupancy histograms.
    Ideal: 0.0
    """
    ctx = _resolve_diagnostic_context(data, **kwargs)
    P = ctx['P_final']

    if P is None or len(P) == 0 or centroids is None or ref_hist is None:
        return FairResult(1.0, "BALANCE", "Missing context for balance.", raw_data=np.array([]))

    # 1. Assign points to clusters
    d = cdist(P, centroids)
    labels = np.argmin(d, axis=1) # (N,)
    
    # 2. Compute Histogram (Frequency)
    n_clusters = len(centroids)
    hist_p = np.bincount(labels, minlength=n_clusters).astype(float)
    hist_p /= np.sum(hist_p) # Normalize to Probability Mass
    
    # 3. JS Divergence
    # Base 2 gives range [0, 1] for JS
    f_val = float(jensenshannon(hist_p, ref_hist, base=2.0))
    return FairResult(
        value=f_val,
        name="BALANCE",
        description=f"Distribution bias across regions is {f_val:.4f} (JS Divergence).",
        raw_data=hist_p # For balance, the raw data is the occupancy histogram
    )
