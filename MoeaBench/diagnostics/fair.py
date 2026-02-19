# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Clinical Fair Metrics (Physical Layer)
================================================

This module implements the "Fair" / "Clinical" metrics logic.
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

class FairResult(DiagnosticValue):
    """ Specialized result for Physical (Fair) metrics. """
    def report(self, **kwargs) -> str:
        if kwargs.get('markdown', True):
            return f"**{self.name}** (Physical): {self.value:.4f}\n- *Meaning*: {self.description}"
        return f"{self.name} (Physical): {self.value:.4f}\n  Meaning: {self.description}"

def headway(data: Any, ref: Optional[Any] = None, s_k: Optional[float] = None, **kwargs) -> float:
    r"""
    [Smart API] Calculates HEADWAY (Algorithmic Progress over BBox).
    
    Definition: $GD_{95}(P \to GT) / s_{FIT}$
    Meaning: How far P is from GT, in units of s_fit (resolution at K).
    Ideal: 0.0
    """
    P, GT, s_fit, _, _ = _resolve_diagnostic_context(data, ref, s_k, **kwargs)
    
    if P is None or GT is None or len(P) == 0:
        return FairResult(np.inf, "HEADWAY", "No points to evaluate.", raw_data=np.array([]))

    # 1. Compute Distances to GT
    d = cdist(P, GT, metric='euclidean')
    min_d = np.min(d, axis=1) # (N,)
    
    # 2. Robust Metric (Percentile 95)
    gd95 = np.percentile(min_d, 95)
    
    # 3. Normalize by Resolution (Scale Invariance)
    u_vals = min_d / s_fit if s_fit > 1e-12 else min_d
    f_val = float(np.percentile(u_vals, 95))
    
    return FairResult(
        value=f_val,
        name="HEADWAY",
        description=f"Population is {f_val:.2f} resolution-units (s_fit) away from the truth (95th percentile).",
        raw_data=u_vals
    )

def closeness(data: Any, ref: Optional[Any] = None, s_k: Optional[float] = None, **kwargs) -> FairResult:
    """
    [Smart API] Calculates the distribution of normalized distances to GT.
    
    Returns u_j = min_dist(p_j, GT) / s_fit
    """
    P, GT, s_fit, _, _ = _resolve_diagnostic_context(data, ref, s_k, **kwargs)

    if P is None or GT is None or len(P) == 0:
        return FairResult(0.0, "CLOSENESS", "No points to evaluate.", raw_data=np.array([]))
        
    # 1. Compute Distances to GT
    d = cdist(P, GT, metric='euclidean')
    min_d = np.min(d, axis=1) # (N,)
    
    # 2. Normalize by Resolution
    u_vals = min_d / s_fit if s_fit > 1e-12 else min_d
    f_val = float(np.median(u_vals))
    
    return FairResult(
        value=f_val,
        name="CLOSENESS",
        description="Median distance to ground truth manifold.",
        raw_data=u_vals
    )

def coverage(data: Any, ref: Optional[Any] = None, **kwargs) -> float:
    r"""
    [Smart API] Calculates COVERAGE.
    
    Definition: $IGD_{mean}(GT \to P)$
    Meaning: Average distance from ANY point in GT to the nearest point in P.
    Ideal: 0.0 (Complete coverage)
    """
    P, GT, _, _, _ = _resolve_diagnostic_context(data, ref, **kwargs)

    if P is None or GT is None or len(P) == 0:
        return FairResult(np.inf, "COVERAGE", "No points to evaluate.", raw_data=np.array([]))

    # 1. Compute Distances from GT to P
    d = cdist(GT, P, metric='euclidean')
    min_d = np.min(d, axis=1) # (|GT|,)
    
    # 2. Mean (IGD)
    f_val = float(np.mean(min_d))
    return FairResult(
        value=f_val,
        name="COVERAGE",
        description=f"Average distance from target manifold to nearest solution is {f_val:.4f}.",
        raw_data=min_d
    )

def gap(data: Any, ref: Optional[Any] = None, **kwargs) -> float:
    r"""
    [Smart API] Calculates GAP (formerly Density).
    
    Definition: $IGD_{95}(GT \to P)$
    Meaning: The size of the "large holes" in coverage (ignoring worst 5% outliers).
    Ideal: 0.0
    """
    P, GT, _, _, _ = _resolve_diagnostic_context(data, ref, **kwargs)

    if P is None or GT is None or len(P) == 0:
        return FairResult(np.inf, "GAP", "No points to evaluate.", raw_data=np.array([]))

    # 1. Compute Distances from GT to P
    d = cdist(GT, P, metric='euclidean')
    min_d = np.min(d, axis=1)
    
    # 2. Percentile 95 (Robust Max)
    f_val = float(np.percentile(min_d, 95))
    return FairResult(
        value=f_val,
        name="GAP",
        description=f"Largest hole detected on the manifold is {f_val:.4f}.",
        raw_data=min_d
    )

def regularity(data: Any, ref_distribution: Optional[np.ndarray] = None, **kwargs) -> float:
    r"""
    [Smart API] Calculates REGULARITY (formerly Uniformity).
    
    Definition: $W_1(d_{NN}(P), d_{NN}(U_{ref}))$
    Meaning: Wasserstein distance between the Nearest-Neighbor distribution of P
             and that of a reference distribution (usually from a Uniform lattice).
    Ideal: 0.0
    """
    P, _, _, _, _ = _resolve_diagnostic_context(data, **kwargs)
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

def balance(data: Any, centroids: Optional[np.ndarray] = None, ref_hist: Optional[np.ndarray] = None, **kwargs) -> float:
    r"""
    [Smart API] Calculates BALANCE.
    
    Definition: $D_{JS}(H_P || H_{ref})$
    Meaning: Jensen-Shannon divergence between cluster occupancy histograms.
    Ideal: 0.0
    """
    P, _, _, _, _ = _resolve_diagnostic_context(data, **kwargs)

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
