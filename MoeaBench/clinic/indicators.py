# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
MoeaBench Clinical Indicators (Quality Score System)
===================================================

This module implements the Normalized Clinical Quality Matrix logic.
All metrics return a Quality Score ($Q \in [0,1]$) where:
- 1.00: Indistinguishable from Optimal Uniform Sampling ($U_K$).
- 0.00: Indistinguishable from Random Sampling ($R_K$).

Strictly anchored to Ground Truth baselines.
"""

import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import wasserstein_distance
from typing import Optional, Tuple

def _normalize_quality(raw_val: float, uni50: float, rand50: float) -> float:
    """
    Quality Score Normalization Formula.
    Clips output to [0, 1] and inverts the scale.
    1.0 = Optimal (uni50), 0.0 = Random (rand50).
    """
    denom = rand50 - uni50
    if abs(denom) < 1e-12:
        # Degenerate baseline range
        # If raw is close to uni, return 1.0 (Optimal). Else assume fail (0.0).
        if abs(raw_val - uni50) < 1e-6:
            return 1.0
        return 0.0
        
    # Calculate error in [0, 1]
    error = (raw_val - uni50) / denom
    error_clipped = np.clip(error, 0.0, 1.0)
    
    # Invert to Quality
    return float(1.0 - error_clipped)

def fit_quality(P: np.ndarray, GT: np.ndarray, s_gt: float,
                uni50: float, rand50: float) -> float:
    r"""
    Calculates FIT Quality Score.
    
    Raw Metric: $GD_{95}(P, GT) / s_{GT}$
    """
    # 1. Compute Distances to GT
    d = cdist(P, GT, metric='euclidean')
    min_d = np.min(d, axis=1) # (N,)
    
    # 2. Raw Metric (Normalized by local resolution)
    gd95 = np.percentile(min_d, 95)
    raw = gd95 / s_gt if s_gt > 1e-12 else gd95
    
    return _normalize_quality(raw, uni50, rand50)

def coverage_quality(P: np.ndarray, GT: np.ndarray,
                     uni50: float, rand50: float) -> float:
    r"""
    Calculates COVERAGE Quality Score.
    
    Raw Metric: $IGD_{mean}(P, GT)$
    """
    # 1. Compute Distances from GT to P
    d = cdist(GT, P, metric='euclidean')
    min_d = np.min(d, axis=1) # (|GT|,)
    
    # 2. Raw Metric
    igd_mean = np.mean(min_d)
    
    return _normalize_quality(igd_mean, uni50, rand50)

def density_quality(P: np.ndarray, GT: np.ndarray,
                    uni50: float, rand50: float) -> float:
    r"""
    Calculates DENSITY Quality Score (formerly Gap).
    
    Raw Metric: $IGD_{95}(P, GT)$
    """
    d = cdist(GT, P, metric='euclidean')
    min_d = np.min(d, axis=1)
    
    igd95 = np.percentile(min_d, 95)
    
    return _normalize_quality(igd95, uni50, rand50)

def regularity_quality(P: np.ndarray, U_ref: np.ndarray,
                       uni50: float, rand50: float) -> float:
    r"""
    Calculates REGULARITY Quality Score (formerly Uniformity).
    
    Raw Metric: $W_1(d_{NN}(P), d_{NN}(U_{ref}))$
    """
    # 1. Nearest Neighbors within P (excluding self)
    d_p = cdist(P, P)
    np.fill_diagonal(d_p, np.inf)
    nn_p = np.min(d_p, axis=1)
    
    # 2. Nearest Neighbors within Reference U_ref
    d_u = cdist(U_ref, U_ref)
    np.fill_diagonal(d_u, np.inf)
    nn_u = np.min(d_u, axis=1)
    
    # 3. Wasserstein Distance
    w1 = wasserstein_distance(nn_p, nn_u)
    
    return _normalize_quality(w1, uni50, rand50)

def balance_quality(P: np.ndarray, centroids: np.ndarray, ref_hist: np.ndarray,
                    uni50: float, rand50: float) -> float:
    r"""
    Calculates BALANCE Quality Score.
    
    Raw Metric: $D_{JS}(H_P || H_{ref})$
    """
    # 1. Assign points to clusters
    d = cdist(P, centroids)
    labels = np.argmin(d, axis=1) # (N,)
    
    # 2. Compute Histogram (Frequency)
    n_clusters = len(centroids)
    hist_p = np.bincount(labels, minlength=n_clusters).astype(float)
    hist_p /= np.sum(hist_p) # Normalize to Probability Mass
    
    # 3. JS Divergence
    js = jensenshannon(hist_p, ref_hist, base=2.0)
    
    return _normalize_quality(js, uni50, rand50)
