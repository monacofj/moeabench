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

def compute_q_fit(fair_fit: float, problem: str, k: int) -> float:
    """Computes Q_FIT using linear interpolation."""
    # Ideal = 0.0
    _, rand50 = baselines.get_baseline_values(problem, k, "fit")
    return _compute_q_linear(fair_fit, 0.0, rand50)

def compute_q_coverage(fair_cov: float, problem: str, k: int) -> float:
    """Computes Q_COVERAGE using linear interpolation."""
    # Ideal = Uni50
    uni50, rand50 = baselines.get_baseline_values(problem, k, "coverage")
    return _compute_q_linear(fair_cov, uni50, rand50)

def compute_q_gap(fair_gap: float, problem: str, k: int) -> float:
    """Computes Q_GAP using linear interpolation."""
    uni50, rand50 = baselines.get_baseline_values(problem, k, "gap")
    return _compute_q_linear(fair_gap, uni50, rand50)

def compute_q_regularity(fair_reg: float, problem: str, k: int) -> float:
    """Computes Q_REGULARITY using linear interpolation."""
    uni50, rand50 = baselines.get_baseline_values(problem, k, "uniformity")
    return _compute_q_linear(fair_reg, uni50, rand50)

def compute_q_balance(fair_bal: float, problem: str, k: int) -> float:
    """Computes Q_BALANCE using linear interpolation."""
    uni50, rand50 = baselines.get_baseline_values(problem, k, "balance")
    return _compute_q_linear(fair_bal, uni50, rand50)

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
    
    # 3. Vectorized Linear
    denom = rand50 # Ideal is 0.0
    if denom <= 1e-12:
         return np.where(fair_vals <= 1e-12, 1.0, 0.0)
         
    error_score = fair_vals / denom
    return 1.0 - np.clip(error_score, 0.0, 1.0)
