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
from typing import Optional, Tuple

def compute_fair_fit(P: np.ndarray, GT: np.ndarray, s_gt: float) -> float:
    r"""
    Calculates FAIR_FIT (Convergence).
    
    Definition: $GD_{95}(P \to GT) / s_{GT}$
    Meaning: How far P is from GT, in units of GT resolution.
    Ideal: 0.0
    """
    if len(P) == 0:
        return np.inf

    # 1. Compute Distances to GT
    d = cdist(P, GT, metric='euclidean')
    min_d = np.min(d, axis=1) # (N,)
    
    # 2. Robust Metric (Percentile 95)
    gd95 = np.percentile(min_d, 95)
    
    # 3. Normalize by Resolution (Scale Invariance)
    # If s_gt is effectively zero, return raw gd95 (though this implies singular GT)
    if s_gt > 1e-12:
        return float(gd95 / s_gt)
    return float(gd95)

def compute_fair_coverage(P: np.ndarray, GT: np.ndarray) -> float:
    r"""
    Calculates FAIR_COVERAGE.
    
    Definition: $IGD_{mean}(GT \to P)$
    Meaning: Average distance from ANY point in GT to the nearest point in P.
    Ideal: 0.0 (Complete coverage)
    """
    if len(P) == 0:
        return np.inf

    # 1. Compute Distances from GT to P
    d = cdist(GT, P, metric='euclidean')
    min_d = np.min(d, axis=1) # (|GT|,)
    
    # 2. Mean (IGD)
    return float(np.mean(min_d))

def compute_fair_gap(P: np.ndarray, GT: np.ndarray) -> float:
    r"""
    Calculates FAIR_GAP (formerly Density).
    
    Definition: $IGD_{95}(GT \to P)$
    Meaning: The size of the "large holes" in coverage (ignoring worst 5% outliers).
    Ideal: 0.0
    """
    if len(P) == 0:
        return np.inf

    # 1. Compute Distances from GT to P
    d = cdist(GT, P, metric='euclidean')
    min_d = np.min(d, axis=1)
    
    # 2. Percentile 95 (Robust Max)
    return float(np.percentile(min_d, 95))

def compute_fair_regularity(P: np.ndarray, U_ref: np.ndarray) -> float:
    r"""
    Calculates FAIR_REGULARITY (formerly Uniformity).
    
    Definition: $W_1(d_{NN}(P), d_{NN}(U_{ref}))$
    Meaning: Wasserstein distance between the Nearest-Neighbor distribution of P
             and that of a reference Uniform lattice U_ref.
    Ideal: 0.0 (Same geometric spacing distribution)
    """
    if len(P) < 2 or len(U_ref) < 2:
        return 0.0 # Degenerate case

    # 1. Nearest Neighbors within P (excluding self)
    d_p = cdist(P, P)
    np.fill_diagonal(d_p, np.inf)
    nn_p = np.min(d_p, axis=1)
    
    # 2. Nearest Neighbors within Reference U_ref
    d_u = cdist(U_ref, U_ref)
    np.fill_diagonal(d_u, np.inf)
    nn_u = np.min(d_u, axis=1)
    
    # 3. Wasserstein Distance
    return float(wasserstein_distance(nn_p, nn_u))

def compute_fair_balance(P: np.ndarray, centroids: np.ndarray, ref_hist: np.ndarray) -> float:
    r"""
    Calculates FAIR_BALANCE.
    
    Definition: $D_{JS}(H_P || H_{ref})$
    Meaning: Jensen-Shannon divergence between cluster occupancy histograms.
    Ideal: 0.0 (Matches reference mass distribution)
    """
    if len(P) == 0:
        return 1.0 # Max divergence

    # 1. Assign points to clusters
    d = cdist(P, centroids)
    labels = np.argmin(d, axis=1) # (N,)
    
    # 2. Compute Histogram (Frequency)
    n_clusters = len(centroids)
    hist_p = np.bincount(labels, minlength=n_clusters).astype(float)
    hist_p /= np.sum(hist_p) # Normalize to Probability Mass
    
    # 3. JS Divergence
    # Base 2 gives range [0, 1] for JS
    return float(jensenshannon(hist_p, ref_hist, base=2.0))
