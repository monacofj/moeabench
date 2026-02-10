# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Clinical Quality Scores (Engineering Layer)
=====================================================

This module implements the "Q-Score" logic ($Q \in [0, 1]$).
Logic:
1.  High-is-Better (1.0 = Optimal, 0.0 = Random/Fail).
12. Formula: $q = 1.0 - clip( (F_rand(fair) - F_rand(ideal)) / (F_rand(rand50) - F_rand(ideal)) )$.
13. Strict Baseline Rules:
    - FIT: ideal=0.0 (Perfect physical match), random=BBox-Random.
    - OTHERS: ideal=Uni50 (FPS of GT), random=Rand50 (Random Subset of GT).
    - BASELINE: Uses Empirical CDF (ECDF) of 200 random samples.
"""

import numpy as np
from . import baselines

def _ecdf(sorted_vals: np.ndarray, x: float) -> float:
    """Computes Empirical CDF value for x given sorted_vals."""
    # side='right': P(X <= x)
    return np.searchsorted(sorted_vals, x, side='right') / len(sorted_vals)

def _compute_q_ecdf(fair_val: float, ideal: float, rand50: float, rand_ecdf: np.ndarray) -> float:
    """
    ECDF-based Q-Score formula.
    
    q = 1 - (F(fair) - F(ideal)) / (F(rand50) - F(ideal))
    """
    f_fair = _ecdf(rand_ecdf, fair_val)
    f_ideal = _ecdf(rand_ecdf, ideal)
    f_rand = _ecdf(rand_ecdf, rand50) 
    
    denom = f_rand - f_ideal
    
    # Safety: If baseline is degenerate (ideal and rand are indistinguishable in distribution)
    # Fail-Closed per specification
    if denom <= 1e-12:
         raise baselines.UndefinedBaselineError(f"Degenerate baseline distribution: F(rand)={f_rand} <= F(ideal)={f_ideal}")

    num = f_fair - f_ideal
    
    # Error Score (0=Ideal, 1=Random)
    error_score = num / denom
    
    # Clip and Invert
    return float(1.0 - np.clip(error_score, 0.0, 1.0))

def compute_q_fit(fair_fit: float, problem: str, k: int) -> float:
    """Computes Q_FIT using ECDF."""
    # Ideal = 0.0
    _, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k, "fit")
    return _compute_q_ecdf(fair_fit, 0.0, rand50, rand_ecdf)

def compute_q_coverage(fair_cov: float, problem: str, k: int) -> float:
    """Computes Q_COVERAGE using ECDF."""
    # Ideal = Uni50
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k, "coverage")
    return _compute_q_ecdf(fair_cov, uni50, rand50, rand_ecdf)

def compute_q_gap(fair_gap: float, problem: str, k: int) -> float:
    """Computes Q_GAP using ECDF."""
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k, "gap")
    return _compute_q_ecdf(fair_gap, uni50, rand50, rand_ecdf)

def compute_q_regularity(fair_reg: float, problem: str, k: int) -> float:
    """Computes Q_REGULARITY using ECDF."""
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k, "uniformity")
    return _compute_q_ecdf(fair_reg, uni50, rand50, rand_ecdf)

def compute_q_balance(fair_bal: float, problem: str, k: int) -> float:
    """Computes Q_BALANCE using ECDF."""
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k, "balance")
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
    _, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k, "fit")
    
    # 3. Vectorized ECDF
    n = len(rand_ecdf)
    f_fair = np.searchsorted(rand_ecdf, fair_vals, side='right') / n
    f_ideal = np.searchsorted(rand_ecdf, 0.0, side='right') / n
    f_rand = np.searchsorted(rand_ecdf, rand50, side='right') / n
    
    denom = f_rand - f_ideal
    if denom <= 1e-12:
         raise baselines.UndefinedBaselineError(f"Degenerate baseline in vector calc: F(rand)={f_rand} <= F(ideal)={f_ideal}")
         
    error_score = (f_fair - f_ideal) / denom
    return 1.0 - np.clip(error_score, 0.0, 1.0)
