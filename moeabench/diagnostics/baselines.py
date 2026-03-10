# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
moeabench Clinical Baselines (Offline & Deterministic)
======================================================

This module handles:
1. Loading offline-computed baselines (Fail-Closed logic).
2. Generating deterministic reference objects ($U_K$, Clusters).
"""

import os
import json
import warnings
import hashlib
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2
from contextlib import contextmanager

# Path to the Authorized Offline Baselines
def _get_default_baseline_path() -> str:
    from ..system import version
    ver = version()
    # Canonical version-specific path
    return os.path.join(os.path.dirname(__file__), f"resources/baselines_v{ver}.json")

BASELINE_JSON_PATH = _get_default_baseline_path()
_CACHE = None
_REGISTERED_SOURCES = [] # List of dicts or paths

class ReproducibilityWarning(UserWarning):
    """Issued when a baseline environment mismatch is detected."""
    pass

class UndefinedBaselineError(Exception):
    """Raised when no technical baseline is found for the requested context."""
    pass

def _verify_baseline_dna(data: Dict[str, Any], source_name: str = "Primary", target_m: Optional[int] = None):
    """Internal helper to verify Environment DNA in baseline JSON."""
    from ..system import version as lib_version
    b_ver = data.get("version")
    b_py = data.get("python_version")
    b_np = data.get("numpy_version")
    b_schema = data.get("schema")
    
    # Schema Check (Fail-Closed)
    if b_schema != "baselines_v4_ecdf" and b_schema is not None:
         raise UndefinedBaselineError(f"Incompatible baseline schema in {source_name}: {b_schema}")

    # Dimension Check (Fail-Closed for MaOP consistency)
    b_m = data.get("mop_dimension")
    if target_m is not None and b_m is not None and b_m != target_m:
         raise UndefinedBaselineError(f"Dimension mismatch in {source_name}: baseline is M={b_m}, expected M={target_m}")

    # Version Check (Warning only)
    if b_ver and b_ver != lib_version():
        warnings.warn(
            f"MoeaBench Version Mismatch ({source_name}): Baseline is {b_ver}, library is {lib_version()}. "
            "Numerical scores may differ across major/minor versions.",
            ReproducibilityWarning
        )

    # Environment DNA Check
    import sys
    import numpy as np
    sys_py = sys.version.split()[0]
    sys_np = np.__version__
    
    if b_py and b_py != sys_py:
        warnings.warn(
            f"Python Environment Shift ({source_name}): Baseline was calibrated on {b_py}, system is {sys_py}. "
            "Bit-for-bit RNG reproducibility is not guaranteed.",
            ReproducibilityWarning
        )
    if b_np and b_np != sys_np:
        warnings.warn(
            f"NumPy Environment Shift ({source_name}): Baseline was calibrated on {b_np}, system is {sys_np}. "
            "RNG implementation shifts may affect diagnostic consistency.",
            ReproducibilityWarning
        )

# Track if cache is stale
_CACHE_DIRTY = True

def register_baselines(source: Union[str, Dict[str, Any]]) -> None:
    """
    Registers a new source of baselines.
    Source can be a path to a JSON file or a dictionary.
    """
    global _CACHE_DIRTY
    _REGISTERED_SOURCES.append(source)
    # Mark cache as dirty to force re-merge on next load
    _CACHE_DIRTY = True

def reset_baselines() -> None:
    """
    Clears all registered sources and resets the cache to the library default.
    """
    global _CACHE, _CACHE_DIRTY, _REGISTERED_SOURCES
    _REGISTERED_SOURCES = []
    _CACHE = None
    _CACHE_DIRTY = True

@contextmanager
def use_baselines(source: Union[str, Dict[str, Any]]):
    """
    Context manager to temporarily use a specific baseline source.
    Useful for longitudinal studies or comparing against historical references.
    """
    global _CACHE, _CACHE_DIRTY, _REGISTERED_SOURCES
    old_sources = list(_REGISTERED_SOURCES)
    old_cache = _CACHE
    old_dirty = _CACHE_DIRTY
    
    try:
        reset_baselines()
        register_baselines(source)
        yield
    finally:
        _REGISTERED_SOURCES = old_sources
        _CACHE = old_cache
        _CACHE_DIRTY = old_dirty

def load_offline_baselines(target_m: Optional[int] = None) -> Dict[str, Any]:
    """
    Loads and merges all offline baseline sources.
    Uses Fail-Closed logic for the primary resource.
    """
    global _CACHE, _CACHE_DIRTY
    
    # Return valid cache if available and not dirty
    # If target_m is provided, we still need to verify the cache's dimension
    if _CACHE is not None and not _CACHE_DIRTY:
        return _CACHE
    
    # 1. Start with the core library baselines
    if not os.path.exists(BASELINE_JSON_PATH):
        raise UndefinedBaselineError(f"Primary baseline file not found at {BASELINE_JSON_PATH}")

    try:
        with open(BASELINE_JSON_PATH, "r") as f:
            full_data = json.load(f)
            
        # 1.1 Environment DNA/Compatibility Verification
        _verify_baseline_dna(full_data, "Primary", target_m=target_m)

        # 1.5 Populate GT registry from primary data if present
        if "problems" in full_data:
            if "_gt_registry" not in full_data:
                full_data["_gt_registry"] = {}
            for prob_id, prob_data in full_data["problems"].items():
                if "gt_reference" in prob_data:
                    full_data["_gt_registry"][prob_id] = prob_data["gt_reference"]
    except UndefinedBaselineError:
        raise
    except Exception as e:
        raise UndefinedBaselineError(f"Failed to parse primary baseline file: {e}")
    
    # 2. Merge registered sources
    for src in _REGISTERED_SOURCES:
        src_data = None
        if isinstance(src, str):
            if os.path.exists(src):
                try:
                    with open(src, "r") as f:
                        src_data = json.load(f)
                        _verify_baseline_dna(src_data, os.path.basename(src), target_m=target_m)
                except UndefinedBaselineError as e:
                    # Specific dimension mismatch is fatal for this source
                    continue 
                except Exception as e:
                    continue
        elif isinstance(src, dict):
            src_data = src
        
        if src_data and "problems" in src_data:
            # Shallow merge of problem dictionaries
            for prob_id, prob_data in src_data["problems"].items():
                full_data["problems"][prob_id] = prob_data
                
                # Also capture gt_reference if present (Plugin Sidecar)
                if "gt_reference" in src_data:
                    # In sidecars, name is often at top level but gt_reference too.
                    # We store it in a hidden index for utils.py to find
                    if "_gt_registry" not in full_data:
                        full_data["_gt_registry"] = {}
                    full_data["_gt_registry"][prob_id] = src_data["gt_reference"]
                elif "gt_reference" in prob_data:
                     # Nested GT
                    if "_gt_registry" not in full_data:
                        full_data["_gt_registry"] = {}
                    full_data["_gt_registry"][prob_id] = prob_data["gt_reference"]

    _CACHE = full_data
    _CACHE_DIRTY = False
    return _CACHE

def get_baseline_values(problem: str, k: int, metric: str, target_m: Optional[int] = None) -> Tuple[float, float]:
    """
    Retrieves (uni50, rand50) for a given (problem, k, metric).
    """
    try:
        bases = load_offline_baselines(target_m=target_m)
    except UndefinedBaselineError:
        raise

    # Structure: problems -> {problem} -> {k} -> {metric} -> {uni50, rand50}
    try:
        p_data = bases.get("problems", {}).get(problem, {})
        
        # Fuzzy K-match logic for local sidecars
        if str(k) in p_data:
            k_data = p_data[str(k)]
        else:
            # Try to find the closest k available in p_data
            available_ks = [int(x) for x in p_data.keys() if x.isdigit()]
            if not available_ks:
                raise ValueError(f"No K-data for problem {problem}")
            closest_k = min(available_ks, key=lambda x: abs(x - k))
            # Only allow a small drift (e.g. 5%) for clinical validity
            if abs(closest_k - k) / max(1, k) > 0.1:
                raise ValueError(f"K={k} too far from closest baseline K={closest_k}")
            k_data = p_data[str(closest_k)]
            
        m_data = k_data.get(metric, {})
        
        uni = m_data.get("uni50")
        rand = m_data.get("rand50")
        
        if uni is None or rand is None:
            raise ValueError("Values are None")
            
        return float(uni), float(rand)
        
    except (AttributeError, ValueError):
        raise UndefinedBaselineError(f"Missing baseline for {problem}, K={k}, {metric}")

def get_baseline_ecdf(problem: str, k: int, metric: str, target_m: Optional[int] = None) -> Tuple[float, float, np.ndarray]:
    """
    Retrieves (uni50, rand50, rand_ecdf) for a given (problem, k, metric).
    """
    try:
        bases = load_offline_baselines(target_m=target_m)
    except UndefinedBaselineError:
        raise

    try:
        p_data = bases.get("problems", {}).get(problem, {})
        
        # Fuzzy K-match logic
        if str(k) in p_data:
            k_data = p_data[str(k)]
        else:
            available_ks = [int(x) for x in p_data.keys() if x.isdigit()]
            if not available_ks:
                raise ValueError(f"No K-data for problem {problem}")
            closest_k = min(available_ks, key=lambda x: abs(x - k))
            if abs(closest_k - k) / max(1, k) > 0.1:
                raise ValueError(f"K={k} too far from closest baseline K={closest_k}")
            k_data = p_data[str(closest_k)]
            
        m_data = k_data.get(metric, {})
        
        uni = m_data.get("uni50")
        rand = m_data.get("rand50")
        rand_ecdf = m_data.get("rand_ecdf")
        
        if uni is None or rand is None or rand_ecdf is None:
            raise UndefinedBaselineError(f"Incomplete baseline for {problem}, K={k}, {metric}")
            
        rand_ecdf = np.array(rand_ecdf, dtype=float)
        
        # Strict Validation
        if len(rand_ecdf) != 200:
            raise UndefinedBaselineError(f"Invalid ECDF length {len(rand_ecdf)} for {problem}, K={k} (Expected 200)")
            
        # Check sorted (non-decreasing)
        if np.any(np.diff(rand_ecdf) < -1e-12): # Allow epsilon noise
             if np.any(np.diff(rand_ecdf) < 0):
                raise UndefinedBaselineError(f"ECDF not sorted for {problem}, K={k}")
        
        # Check Median Consistency (Optional but good)
        calc_median = np.median(rand_ecdf)
        if not np.isclose(rand, calc_median, atol=0.2): # Loosened for downsampled ECDFs
             # Log warning or fail? Fail-Closed means fail.
             # But floating point issues / downsampling might occur. tolerance 0.2 is safe for 200/1000 samples.
             raise UndefinedBaselineError(f"Baseline mismatch: rand50 ({rand}) != median(ecdf) ({calc_median}) for {problem}, K={k}")
            
        return float(uni), float(rand), rand_ecdf
        
    except (AttributeError, ValueError) as e:
        raise UndefinedBaselineError(f"Invalid baseline data for {problem}, K={k}: {e}")


# Internal cache for Reference Sets (FPS)
_FPS_CACHE = {}

def clear_baselines_cache():
    global _FPS_CACHE
    _FPS_CACHE = {}

def get_ref_uk(gt: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    Generates the Deterministic Reference $U_K$ (Uniform-at-K).
    Logic: Farthest Point Sampling (FPS) subset of GT.
    Optimized with internal caching.
    """
    if len(gt) <= k:
        return gt
        
    # Enforce hash-based key for FPS to avoid collisions across problems
    # but keep id(gt) for speed if it's the same object reference
    data_hash = hashlib.sha256(gt.tobytes()).hexdigest()
    key = (data_hash, k, seed)
    
    # Check cache
    if key in _FPS_CACHE:
        cached_indices = _FPS_CACHE[key]
        if len(cached_indices) >= k:
            return gt[cached_indices[:k]]
    else:
        cached_indices = []

    # Resume FPS or start fresh
    rng = np.random.RandomState(seed)
    
    if not cached_indices:
        start_idx = rng.randint(0, len(gt))
        cached_indices = [start_idx]
        min_dists = cdist(gt[start_idx:start_idx+1], gt, metric='euclidean')[0]
    else:
        # Re-derive min_dists for the current set
        # This is slightly expensive but only happens once when expanding K
        points = gt[cached_indices]
        d = cdist(points, gt, metric='euclidean')
        min_dists = np.min(d, axis=0)

    # Grow to target k or a reasonable buffer (e.g. 200)
    target_k = max(k, 200)
    target_k = min(target_k, len(gt))
    
    for _ in range(len(cached_indices), target_k):
        next_idx = np.argmax(min_dists)
        cached_indices.append(next_idx)
        
        # Update distances
        new_dists = cdist(gt[next_idx:next_idx+1], gt, metric='euclidean')[0]
        min_dists = np.minimum(min_dists, new_dists)
        
    _FPS_CACHE[key] = cached_indices
    return gt[cached_indices[:k]]

def get_ref_clusters(gt: np.ndarray, c: int = 32, seed: int = 0) -> Any:
    """
    Generates the Deterministic Reference Clusters (Centroids).
    
    Logic: K-Means on GT with fixed seed.
    Returns: (centroids, labels)
    """
    # Normalize GT to [0,1] for clustering stability if not already? 
    # Assuming GT is pre-normalized or typically [0,1] box.
    # Using scipy.cluster.vq.kmeans2
    n_pts = len(gt)
    if n_pts < c:
        # Fallback if GT is tiny: just use GT points as centroids
        return gt, np.arange(n_pts)
        
    with warnings.catch_warnings():
        # SciPy may warn about empty clusters if data is highly collapsed or 
        # for specific seeds. moeabench handles histograms with zeros gracefully.
        warnings.simplefilter("ignore", category=UserWarning)
        centroids, labels = kmeans2(gt, c, minit='points', seed=seed, iter=20)
    return centroids, labels

def get_resolution_factor(gt: np.ndarray) -> float:
    """
    Calculates s_GT: Median Nearest Neighbor distance within GT.
    Used for normalizing HEADWAY/FIT.
    """
    if len(gt) < 2:
        return 1.0 # Fallback
        
    d = cdist(gt, gt)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d, axis=1)
    return float(np.median(min_d))

def get_resolution_factor_k(gt: np.ndarray, k: int, seed: int = 0) -> float:
    """
    Calculates s_K: Median Nearest Neighbor distance within U_K (FPS of GT).
    Used for normalizing HEADWAY/FIT in finite sampling regimes.
    """
    u_ref = get_ref_uk(gt, k, seed=seed)
    if len(u_ref) < 2:
        return 1.0
        
    d = cdist(u_ref, u_ref)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d, axis=1)
    return float(np.median(min_d))

def snap_k(k_raw: int, problem: Optional[str] = None) -> int:
    """
    Snaps the raw population size K to the nearest supported baseline K
    using a strict 'floor' logic for consistency with the audit protocol.
    
    If 'problem' is provided, it first checks the registered baselines (including sidecars)
    to see what K values are actually available for that problem.
    """
    if problem:
        try:
            bases = load_offline_baselines()
            p_data = bases.get("problems", {}).get(problem, {})
            available_ks = [int(x) for x in p_data.keys() if x.isdigit()]
            if available_ks:
                # Return closest K among available ones
                return min(available_ks, key=lambda x: abs(x - k_raw))
        except:
            pass

    if k_raw < 10:
        return 10
    if k_raw < 50:
        return k_raw
    
    grid = [50, 100, 150, 200]
    # Filter grid items <= k_raw
    candidates = [g for g in grid if g <= k_raw]
    if not candidates:
        # Should not happen given k_raw >= 50 check, but safe fallback
        return 50
    return max(candidates)
