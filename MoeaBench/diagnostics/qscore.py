# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Clinical Quality Scores (Engineering Layer)
=====================================================

This module implements the "Q-Score" logic ($Q \in [0, 1]$).
Logic:
1.  High-is-Better (1.0 = Optimal, 0.0 = Random/Fail).
2.  Formula: $q = 1.0 - clip( (fair - ideal) / (random - ideal) )$.
3.  Strict Baseline Rules:
    - FIT: ideal=0.0 (Perfect physical match), random=BBox-Random.
    - OTHERS: ideal=Uni50 (FPS of GT), random=Rand50 (Random Subset of GT).
"""

import numpy as np
from . import baselines

def _compute_q_generic(fair_val: float, ideal: float, random: float) -> float:
    """
    Generic Q-Score formula.
    """
    denom = random - ideal
    
    # Safety: If denom is tiny (degenerate baseline range)
    if abs(denom) < 1e-12:
        # If fair is physically close to ideal, score 1.0. Else 0.0.
        if abs(fair_val - ideal) < 1e-6:
            return 1.0
        return 0.0
        
    # Standard Normalization
    # error_score in [0, 1] where 0=ideal, 1=random
    error_score = (fair_val - ideal) / denom
    
    # Clip to [0, 1] to handle super-optimal or super-bad values
    error_clipped = np.clip(error_score, 0.0, 1.0)
    
    # Invert to Quality (1.0 = Ideal)
    return float(1.0 - error_clipped)

def compute_q_fit(fair_fit: float, problem: str, k: int) -> float:
    """
    Computes Q_FIT.
    Rule: Ideal = 0.0 (Physical Perfection).
          Random = Baseline 'fit' (contains BBox-Random).
    """
    # Fetch random baseline. Ignore uni50 from JSON (enforce 0.0).
    _, rand = baselines.get_baseline_values(problem, k, "fit")
    return _compute_q_generic(fair_fit, 0.0, rand)

def compute_q_coverage(fair_cov: float, problem: str, k: int) -> float:
    """
    Computes Q_COVERAGE.
    Rule: Ideal = Uni50, Random = Rand50.
    """
    uni, rand = baselines.get_baseline_values(problem, k, "coverage")
    return _compute_q_generic(fair_cov, uni, rand)

def compute_q_gap(fair_gap: float, problem: str, k: int) -> float:
    """
    Computes Q_GAP.
    Rule: Ideal = Uni50, Random = Rand50.
    """
    uni, rand = baselines.get_baseline_values(problem, k, "gap")
    return _compute_q_generic(fair_gap, uni, rand)

def compute_q_regularity(fair_reg: float, problem: str, k: int) -> float:
    """
    Computes Q_REGULARITY.
    Rule: Ideal = Uni50, Random = Rand50.
    """
    uni, rand = baselines.get_baseline_values(problem, k, "uniformity")
    return _compute_q_generic(fair_reg, uni, rand)

def compute_q_balance(fair_bal: float, problem: str, k: int) -> float:
    """
    Computes Q_BALANCE.
    Rule: Ideal = Uni50, Random = Rand50.
    """
    uni, rand = baselines.get_baseline_values(problem, k, "balance")
    return _compute_q_generic(fair_bal, uni, rand)
