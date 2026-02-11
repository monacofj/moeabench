# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Clinical Quality Scores (Engineering Layer)
=====================================================

This module implements the "Q-Score" logic ($Q \in [0, 1]$).
Logic:
1.  High-is-Better (1.0 = Optimal, 0.0 = Random/Fail).
2.  Formula: Linear interpolation between Ideal (Q=1) and Random (Q=0).
3.  Strict Baseline Rules:
    - FIT: ideal=0.0 (Perfect physical match), random=BBox-Random.
    - OTHERS: ideal=Uni50 (FPS of GT), random=Rand50 (Random Subset of GT).
"""

import numpy as np
from . import baselines

def _compute_q_linear(fair_val: float, ideal: float, rand50: float) -> float:
    """
    Linear Q-Score formula.
    
    q = 1 - clip( (fair - ideal) / (rand50 - ideal) )
    """
    denom = rand50 - ideal
    
    # Safety: If baseline is degenerate (ideal and rand are too close)
    if denom <= 1e-12:
         return 1.0 if fair_val <= ideal + 1e-12 else 0.0

    num = fair_val - ideal
    
    # Error Score (0=Ideal, 1=Random)
    error_score = num / denom
    
    # Clip and Invert
    return float(1.0 - np.clip(error_score, 0.0, 1.0))

def _compute_q_ecdf(fair_val: float, ideal: float, rand50: float, rand_ecdf: np.ndarray) -> float:
    """
    ECDF Q-Score formula.
    
    q = 1 - clip( (F(fair) - F(ideal)) / (F(rand50) - F(ideal)) )
    """
    # 1. Compute Cumulative Probabilities (F)
    # np.searchsorted(sorted_array, value) returns index where value should be inserted
    # Probability P(X < value) ~ index / N
    N = len(rand_ecdf)
    
    idx_fair = np.searchsorted(rand_ecdf, fair_val, side='right')
    F_fair = idx_fair / N
    
    idx_ideal = np.searchsorted(rand_ecdf, ideal, side='right')
    F_ideal = idx_ideal / N
    
    idx_rand = np.searchsorted(rand_ecdf, rand50, side='right')
    F_rand = idx_rand / N
    
    # 2. Denominator: Range between Ideal and Random in Probability Space
    denom = F_rand - F_ideal
    
    # Safety: If baseline is degenerate (F_rand == F_ideal)
    # This happens if ideal and rand50 fall in the same gap or plateau of ECDF.
    if denom <= 1e-12:
         # If fair is physically close to ideal (within epsilon), it's perfect.
         # Otherwise it's worse (0.0), assuming ideal is the target.
         return 1.0 if abs(fair_val - ideal) <= 1e-9 else 0.0

    # 3. Interpolate
    num = F_fair - F_ideal
    error_score = num / denom
    
    return float(1.0 - np.clip(error_score, 0.0, 1.0))

def compute_q_fit(fair_fit: float, problem: str, k: int) -> float:
    """Computes Q_FIT using strict linear baseline (Ideal -> Rand50)."""
    # Ideal = 0.0 (Perfect physical match)
    _, rand50 = baselines.get_baseline_values(problem, k, "fit")
    return _compute_q_linear(fair_fit, 0.0, rand50)

def compute_q_coverage(fair_cov: float, problem: str, k: int) -> float:
    """Computes Q_COVERAGE using ECDF."""
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k, "cov")
    return _compute_q_ecdf(fair_cov, uni50, rand50, rand_ecdf)

def compute_q_gap(fair_gap: float, problem: str, k: int) -> float:
    """Computes Q_GAP using ECDF."""
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k, "gap")
    return _compute_q_ecdf(fair_gap, uni50, rand50, rand_ecdf)

def compute_q_regularity(fair_reg: float, problem: str, k: int) -> float:
    """Computes Q_REGULARITY using ECDF."""
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k, "reg")
    return _compute_q_ecdf(fair_reg, uni50, rand50, rand_ecdf)

def compute_q_balance(fair_bal: float, problem: str, k: int) -> float:
    """Computes Q_BALANCE using ECDF."""
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k, "bal")
    return _compute_q_ecdf(fair_bal, uni50, rand50, rand_ecdf)

def compute_q_fit_points(dists: np.ndarray, problem: str, k: int, s_fit: float) -> np.ndarray:
    """
    Vectorized Q-Score calculation for point cloud (FIT).
    
    Args:
        dists: Array of raw distances (min d(P->GT))
        problem: Problem name
        k: K value
        s_fit: Resolution factor for normalization
        
    Returns:
        np.ndarray: Array of Q-Scores for each point (0..1)
    """
    # 1. Normalize
    fair_vals = dists / s_fit
    
    # 2. Get Baseline
    _, rand50 = baselines.get_baseline_values(problem, k, "fit")
    
    # 3. Vectorized ECDF
    _, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k, "fit")
    N = len(rand_ecdf)
    
    # searchsorted is vectorized for the 'v' argument (fair_vals)
    idx_fair = np.searchsorted(rand_ecdf, fair_vals, side='right')
    F_fair = idx_fair / N
    
    # Scalar lookups for Ideal/Rand
    idx_ideal = np.searchsorted(rand_ecdf, 0.0, side='right')
    F_ideal = idx_ideal / N
    
    idx_rand = np.searchsorted(rand_ecdf, rand50, side='right')
    F_rand = idx_rand / N
    
    denom = F_rand - F_ideal
    
    if denom <= 1e-12:
         # ideal ~ rand
         return np.where(F_fair <= F_ideal + 1e-12, 1.0, 0.0)
         
    num = F_fair - F_ideal
    error_score = num / denom
    
    return 1.0 - np.clip(error_score, 0.0, 1.0)
