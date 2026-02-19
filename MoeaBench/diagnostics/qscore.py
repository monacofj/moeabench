# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Clinical Quality Scores (Engineering Layer)
=====================================================

This module implements the "Q-Score" logic ($Q \\in [0, 1]$).
Logic:
1.  High-is-Better (1.0 = Optimal, 0.0 = Random/Fail).
2.  Formula: Linear interpolation between Ideal (Q=1) and Random (Q=0).
3.  Strict Baseline Rules:
    - HEADWAY: ideal=0.0 (Better-than-noise progress), random=BBox-Random.
    - OTHERS: ideal=Uni50 (FPS of GT), random=Rand50 (Random Subset of GT).
"""

import numpy as np
from scipy.spatial.distance import cdist
from . import baselines
from .utils import _resolve_diagnostic_context
from . import fair
from .base import DiagnosticValue
from typing import Any, Optional

class QResult(DiagnosticValue):
    """ Specialized result for Clinical (Q-Score) metrics. """
    
    _LABELS = {
        "Q_HEADWAY":   {0.95: "Near-Ideal Progress", 0.85: "Strong Progress", 0.67: "Effective", 0.34: "Partial", 0.0: "Static/Random"},
        "Q_CLOSENESS": {0.95: "Asymptotic", 0.85: "High Precision", 0.67: "Sufficient", 0.34: "Coarse", 0.0: "Remote"},
        "Q_COVERAGE":  {0.95: "Exhaustive", 0.85: "Extensive", 0.67: "Standard", 0.34: "Limited", 0.0: "Collapsed"},
        "Q_GAP":       {0.95: "High Continuity", 0.85: "Stable", 0.67: "Managed Gaps", 0.34: "Interrupted", 0.0: "Fragmented"},
        "Q_REGULARITY":{0.95: "Asymptotic Regularity", 0.85: "Ordered", 0.67: "Consistent", 0.34: "Irregular", 0.0: "Unstructured"},
        "Q_BALANCE":   {0.95: "Near-Ideal Balance", 0.85: "Equitable", 0.67: "Fair", 0.34: "Biased", 0.0: "Skewed"}
    }
    
    @property
    def verdict(self) -> str:
        """Returns the categorical label for this Q-Score."""
        q = float(self.value)
        if self.name in self._LABELS:
            matrix = self._LABELS[self.name]
            for thresh in sorted(matrix.keys(), reverse=True):
                if q >= thresh:
                    return matrix[thresh]
        return "Undefined"
    
    def report(self, **kwargs) -> str:
        q = float(self.value)
        label = "Undefined"
        if self.name in self._LABELS:
            matrix = self._LABELS[self.name]
            for thresh in sorted(matrix.keys(), reverse=True):
                if q >= thresh:
                    label = matrix[thresh]
                    break
        
        if kwargs.get('markdown', True):
            return f"**{self.name}** (Clinical Score): {q:.3f}\n- *Verdict*: {label}\n- *Insight*: {self.description}"
        
        # Clean terminal output: pad name to width if provided
        width = kwargs.get('width', len(self.name))
        name_str = f"{self.name}".ljust(width)
        return f"{name_str} : {q:.3f} [{label}] - {self.description}"


def _wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """Wasserstein-1 distance in 1D (a.k.a. EMD in 1D) without SciPy.

    Both inputs are treated as empirical samples with uniform weights.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.size == 0 or v.size == 0:
        return float('nan')

    u = np.sort(u)
    v = np.sort(v)

    # Merge all sample points; integrate absolute CDF difference.
    all_x = np.concatenate([u, v])
    all_x.sort()
    if all_x.size <= 1:
        return 0.0

    # Empirical CDF values at left of each interval
    iu = np.searchsorted(u, all_x[:-1], side='right') / u.size
    iv = np.searchsorted(v, all_x[:-1], side='right') / v.size
    dx = np.diff(all_x)
    return float(np.sum(np.abs(iu - iv) * dx))


def compute_q_wasserstein(front_samples: np.ndarray, ideal_samples: np.ndarray, rand_samples: np.ndarray, *, eps: float = 1e-12) -> float:
    """Distributional Q-score via Wasserstein-1 (EMD 1D).

    Definition:
        Q = d(F, R) / (d(F, R) + d(F, I))

    where d is Wasserstein-1 on the same scalar FAIR metric.
    """
    f = np.asarray(front_samples, dtype=float)
    i = np.asarray(ideal_samples, dtype=float)
    r = np.asarray(rand_samples, dtype=float)
    if f.size == 0 or i.size == 0 or r.size == 0:
        return float('nan')
    d_i = _wasserstein_1d(f, i)
    d_r = _wasserstein_1d(f, r)
    delta = _wasserstein_1d(i, r)
    
    # Monotonicity Gate: If error >= baseline noise budget, score is 0.
    if d_i >= delta:
        return 0.0
        
    denom = d_i + d_r
    if denom <= eps:
        return 1.0
    return float(d_r / denom)

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


def _compute_q_loglinear(fair_val: float, ideal: float, rand50: float) -> float:
    """Log-linear variant of the Q-score mapping.

    Keeps the same anchors as the linear mapping (Q=1 at `ideal`, Q=0 at
    `rand50`), but expands resolution near Q~1 when `fair_val` is small.

        q = 1 - clip( log1p(fair-ideal) / log1p(rand50-ideal) )

    This is useful when the linear mapping compresses most realistic values
    into Q very close to 1.0.
    """
    denom_raw = rand50 - ideal
    if denom_raw <= 1e-12:
        return 1.0 if fair_val <= ideal + 1e-12 else 0.0

    # Ensure non-negative arguments to log1p.
    num_raw = max(fair_val - ideal, 0.0)
    denom = np.log1p(denom_raw)
    if denom <= 1e-12:
        return 1.0 if num_raw <= 1e-12 else 0.0

    error_score = np.log1p(num_raw) / denom
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
         # Otherwise it's worse (0.0), assuming ideal is the target.
         val = 1.0 if abs(fair_val - ideal) <= 1e-9 else 0.0
         return val

    # 3. Interpolate
    num = F_fair - F_ideal
    error_score = num / denom
    
    return float(1.0 - np.clip(error_score, 0.0, 1.0))

def q_headway(data: Any, ref: Optional[Any] = None, s_k: Optional[float] = None, **kwargs) -> float:
    """[Smart API] Computes Q_HEADWAY using a log-linear baseline (Ideal -> Rand50).
    """
    s_fit = s_k if s_k is not None else 1.0
    if hasattr(data, 'value') and isinstance(data, fair.FairResult):
        f_val = float(data.value)
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    elif isinstance(data, (float, int, np.number)):
        f_val = float(data)
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    else:
        P, GT, s_ctx, problem, k = _resolve_diagnostic_context(data, ref, s_k, **kwargs)
        s_fit = s_ctx
        f_val = float(fair.headway(P, GT, s_fit))
    
    # Snap K to supported baseline grid
    k_snap = baselines.snap_k(k)
    
    # Ideal = 0.0 (Better-than-noise progress)
    _, rand50_raw = baselines.get_baseline_values(problem, k_snap, "headway")
    
    # Normalize baseline to match FAIR units (s_fit)
    rand50 = rand50_raw / s_fit if (s_fit and s_fit > 1e-12) else rand50_raw

    q_val = _compute_q_loglinear(f_val, 0.0, rand50)
    
    return QResult(
        value=q_val,
        name="Q_HEADWAY",
        description="Measures the algorithmic progress (headway) made beyond the random bounding box."
    )

def q_closeness(data: Any, ref: Optional[Any] = None, s_k: Optional[float] = None, **kwargs) -> float:
    """[Smart API] Computes Q_CLOSENESS using GT-normal-blur baseline (W1).
    """
    if hasattr(data, 'raw_data') and data.raw_data is not None:
        # Pre-calculated distribution in a FairResult/DiagnosticValue
        u_dist = data.raw_data
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        # Pre-calculated distribution (raw)
        u_dist = data
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    else:
        P, GT, s_fit, problem, k = _resolve_diagnostic_context(data, ref, s_k, **kwargs)
        res = fair.closeness(P, GT, s_fit)
        u_dist = res.raw_data
    
    if u_dist is None or u_dist.size == 0:
        return QResult(0.0, "Q_CLOSENESS", "Empty distribution (Failed).")
        
    # Snap K to supported baseline grid
    k_snap = baselines.snap_k(k)
    
    # 1. Get Baseline (G_blur ECDF)
    _, _, rand_ecdf = baselines.get_baseline_ecdf(problem, k_snap, "closeness")
    
    # 2. Stability: Explicit ideal samples (all zeros, same size as u_dist)
    ideal_samples = np.zeros_like(u_dist)
    
    # 3. Compute Wasserstein-1 Distances
    d_ideal = _wasserstein_1d(u_dist, ideal_samples)
    d_bad_ideal = _wasserstein_1d(rand_ecdf, ideal_samples)
    
    # 4. Monotonicity Gate
    if d_ideal >= d_bad_ideal:
        return QResult(0.0, "Q_CLOSENESS", "Worse than random baseline.")
        
    d_bad = _wasserstein_1d(u_dist, rand_ecdf)
    denom = d_bad + d_ideal
    q_val = float(d_bad / denom) if denom > 1e-12 else 1.0
    
    return QResult(
        value=q_val,
        name="Q_CLOSENESS",
        description="Measures the precision/sharpness of convergence to the manifold."
    )

def q_coverage(data: Any, ref: Optional[Any] = None, **kwargs) -> float:
    """[Smart API] Computes Q_COVERAGE using ECDF."""
    if hasattr(data, 'value') and isinstance(data, fair.FairResult):
        f_val = float(data.value)
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    elif isinstance(data, (float, int, np.number)):
        f_val = float(data)
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    else:
        P, GT, _, problem, k = _resolve_diagnostic_context(data, ref, **kwargs)
        f_val = float(fair.coverage(P, GT))
        
    k_snap = baselines.snap_k(k)
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k_snap, "cov")
    q_val = _compute_q_ecdf(f_val, uni50, rand50, rand_ecdf)
    return QResult(value=q_val, name="Q_COVERAGE", description="Measures how well the population spans the entire front.")

def q_gap(data: Any, ref: Optional[Any] = None, **kwargs) -> float:
    """[Smart API] Computes Q_GAP using ECDF."""
    if hasattr(data, 'value') and isinstance(data, fair.FairResult):
        f_val = float(data.value)
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    elif isinstance(data, (float, int, np.number)):
        f_val = float(data)
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    else:
        P, GT, _, problem, k = _resolve_diagnostic_context(data, ref, **kwargs)
        f_val = float(fair.gap(P, GT))
        
    k_snap = baselines.snap_k(k)
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k_snap, "gap")
    q_val = _compute_q_ecdf(f_val, uni50, rand50, rand_ecdf)
    return QResult(value=q_val, name="Q_GAP", description="Measures the continuity of the front (lack of large holes).")

def q_regularity(data: Any, ref_distribution: Optional[np.ndarray] = None, **kwargs) -> float:
    """[Smart API] Computes Q_REGULARITY using ECDF."""
    if hasattr(data, 'value') and isinstance(data, fair.FairResult):
        f_val = float(data.value)
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    elif isinstance(data, (float, int, np.number)):
        f_val = float(data)
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    else:
        P, _, _, problem, k = _resolve_diagnostic_context(data, **kwargs)
        f_val = float(fair.regularity(P, ref_distribution))
        
    k_snap = baselines.snap_k(k)
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k_snap, "reg")
    q_val = _compute_q_ecdf(f_val, uni50, rand50, rand_ecdf)
    return QResult(value=q_val, name="Q_REGULARITY", description="Measures the uniformity of point spacing (lattice order).")

def q_balance(data: Any, centroids: Optional[np.ndarray] = None, ref_hist: Optional[np.ndarray] = None, **kwargs) -> float:
    """[Smart API] Computes Q_BALANCE using ECDF."""
    if hasattr(data, 'value') and isinstance(data, fair.FairResult):
        f_val = float(data.value)
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    elif isinstance(data, (float, int, np.number)):
        f_val = float(data)
        problem = kwargs.get('problem', "Unknown")
        k = kwargs.get('k', 100)
    else:
        P, _, _, problem, k = _resolve_diagnostic_context(data, **kwargs)
        f_val = float(fair.balance(P, centroids, ref_hist))
        
    k_snap = baselines.snap_k(k)
    uni50, rand50, rand_ecdf = baselines.get_baseline_ecdf(problem, k_snap, "bal")
    q_val = _compute_q_ecdf(f_val, uni50, rand50, rand_ecdf)
    return QResult(value=q_val, name="Q_BALANCE", description="Measures the equitable mass distribution across the manifold.")

def q_headway_points(data: Any, ref: Optional[Any] = None, s_k: Optional[float] = None, **kwargs) -> np.ndarray:
    """
    [Smart API] Vectorized Q-Score calculation for point cloud (HEADWAY).
    Supports either (P, GT) or pre-calculated dists.
    """
    P, GT, s_fit, problem, k = _resolve_diagnostic_context(data, ref, s_k, **kwargs)
    
    if P is None or len(P) == 0:
        return np.array([])

    # If data was 1D and no GT, assume they are already raw dists
    if P.ndim == 1:
        u_vals = P 
    else:
        if GT is None:
             raise ValueError("Ground truth (ref) required to calculate distances from front.")
        d = cdist(P, GT, metric='euclidean')
        u_vals = np.min(d, axis=1)

    # Apply resolution scaling
    fair_vals = u_vals / s_fit if s_fit > 1e-12 else u_vals
    
    # 2. Get Baseline (Rand50 in FAIR space)
    k_snap = baselines.snap_k(k)
    _, rand50_raw = baselines.get_baseline_values(problem, k_snap, "headway")
    rand50 = rand50_raw / s_fit if (s_fit and s_fit > 1e-12) else rand50_raw

    # 3. Vectorized log-linear mapping
    denom_raw = max(rand50, 0.0)
    denom = np.log1p(denom_raw)
    if denom <= 1e-12:
        return np.where(fair_vals <= 1e-12, 1.0, 0.0)

    num_raw = np.maximum(fair_vals, 0.0)
    error_score = np.log1p(num_raw) / denom
    return 1.0 - np.clip(error_score, 0.0, 1.0)

def q_closeness_points(data: Any, ref: Optional[Any] = None, s_k: Optional[float] = None, **kwargs) -> np.ndarray:
    """
    [Smart API] Vectorized Q-Score calculation for point cloud (CLOSENESS).
    Supports either (P, GT) or pre-calculated dists.
    """
    P, GT, s_fit, problem, k = _resolve_diagnostic_context(data, ref, s_k, **kwargs)

    if P is None or len(P) == 0:
        return np.array([])

    if P.ndim == 1:
        raw_u = P 
    else:
        if GT is None:
             raise ValueError("Ground truth (ref) required to calculate distances from front.")
        d = cdist(P, GT, metric='euclidean')
        raw_u = np.min(d, axis=1)
    
    # Apply resolution scaling
    u_vals = raw_u / s_fit if s_fit > 1e-12 else raw_u
    
    # 2. Get Baseline (Rand50 in FAIR space)
    k_snap = baselines.snap_k(k)
    _, rand50 = baselines.get_baseline_values(problem, k_snap, "closeness")

    # 3. Vectorized log-linear mapping
    denom_raw = max(rand50, 0.0)
    denom = np.log1p(denom_raw)
    if denom <= 1e-12:
        return np.where(u_vals <= 1e-12, 1.0, 0.0)

    num_raw = np.maximum(u_vals, 0.0)
    error_score = np.log1p(num_raw) / denom
    return 1.0 - error_score
