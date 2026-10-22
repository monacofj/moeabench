# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
moeabench Calibration Engine (Automated MOP Profiling)
=====================================================

This module provides the logic to calculate Clinical Baselines (ECDF, Uni50, Rand50)
for any problem instance.
"""

import os
import json
import inspect
import numpy as np
from typing import Optional, List, Dict, Any, Union
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from . import baselines as base
from . import fair
from ..core.base import emit_output

# Defaults aligned with baselines_v4
N_SAMPLES_ECDF = 200
N_IDEAL_DRAWS = 30
K_GRID_DEFAULT = list(range(10, 51)) + [100, 150, 200]
SEED_START = 1000

def independent_analytical_reference(
    mop: Any, gt: np.ndarray, n_fresh: int = 1024,
    sampling_seed: int = 123,
):
    """Representative deterministic subsample of points absent from finite GT.

    Some built-in Pareto-front samplers reset their RNG for every call,
    whereas others traverse an *ordered* objective-space grid. Taking the
    first n_fresh new points is therefore unsafe: it can concentrate all
    samples in one objective-space region (DPF1/2/4 and DTLZ9, issue #75).

    Request an analytical reservoir at least four times larger than the
    desired sample, remove points coinciding with the finite GT, then draw
    n_fresh points without replacement with a *private, fixed-seed* RNG.
    This ensures reproducibility and reduces ordered-prefix bias; it does
    not by itself establish that the underlying PF generator is statistically
    representative. Scientific acceptance requires separate robustness tests.
    The selection seed is independent of the MOP's own internal RNG.
    """
    gt = np.asarray(gt, dtype=float)
    if gt.ndim != 2 or len(gt) < 2 or not np.all(np.isfinite(gt)):
        raise ValueError("GT must be a finite nonempty 2D reference")
    if not isinstance(n_fresh, int) or n_fresh < 2:
        raise ValueError("n_fresh must be an integer >= 2")
    if not isinstance(sampling_seed, int) or sampling_seed < 0:
        raise ValueError("sampling_seed must be a nonnegative integer")

    # Deliberately oversample: prefix reuse is common for deterministic PFs;
    # the independently selected pool must cover the entire returned set.
    requested = len(gt) + 4 * n_fresh
    try:
        candidates = np.asarray(mop.pf(n_points=requested), dtype=float)
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return None
    if (candidates.ndim != 2 or candidates.shape[1] != gt.shape[1]
            or not np.all(np.isfinite(candidates))):
        return None
    distances = cKDTree(gt).query(candidates, k=1)[0]
    threshold = max(1e-12, 1e-10 * float(np.max(np.ptp(gt, axis=0))))
    fresh = candidates[distances > threshold]
    if len(fresh) < n_fresh:
        return None
    choice = np.random.default_rng(sampling_seed).choice(
        len(fresh), size=n_fresh, replace=False
    )
    return fresh[choice]


def estimate_ideal_residue(gt: np.ndarray, independent_ideal: np.ndarray, s_k: float) -> float:
    """Estimate finite-GT residue from independent ideal-front samples.

    Implements ADR 0037: median_{ideal sample} d(x, GT) / s_K.
    A sample composed of GT points is *invalid*, even if a different
    farthest-point-sampling seed was used to select those same points.
    """
    gt = np.asarray(gt, dtype=float)
    independent_ideal = np.asarray(independent_ideal, dtype=float)
    if gt.ndim != 2 or independent_ideal.ndim != 2 or gt.shape[1] != independent_ideal.shape[1]:
        raise ValueError("GT and independent ideal samples require matching objective dimensions")
    if len(gt) < 2 or len(independent_ideal) < 2:
        raise ValueError("At least two points are required for residue estimation")
    if not (np.all(np.isfinite(gt)) and np.all(np.isfinite(independent_ideal))):
        raise ValueError("Ideal reference and GT must contain finite values")
    if not np.isfinite(s_k) or s_k <= 0:
        raise ValueError("s_K must be strictly positive")
    distances = cKDTree(gt).query(independent_ideal, k=1)[0]
    threshold = max(1e-12, 1e-10 * float(np.max(np.ptp(gt, axis=0))))
    if np.any(distances <= threshold):
        raise ValueError("ideal reference overlaps finite GT: independent samples required")
    return float(np.median(distances) / s_k)


def calibrate_mop(mop: Any, 
                  source_baseline: Optional[str] = None, 
                  source_gt: Optional[Union[str, np.ndarray]] = None,
                  source_search: Optional[Any] = None,
                  force: bool = False, 
                  k_values: Optional[List[int]] = None,
                  population_size: Optional[int] = None,
                  size: Optional[int] = None) -> bool:
    """
    Programmatic entry point for MOP calibration.
    
    This function implements the three Ground Truth (GT) protocols:
    A) Analytical (Default): Calls mop.pf(2000) using the ps() mathematical formula.
    B) Externally Supplied (source_gt): Uses a provided CSV or array as the "Truth".
    C) Empirical Search (source_search): Runs a MOEA search to find the GT on-the-fly.
    
    The resulting profile is saved as a 'Sidecar' JSON (source_baseline).
    
    Returns:
        bool: True if recalibrated, False if loaded from existing file.
    """
    mop_name = getattr(mop, 'name', mop.__class__.__name__)
    if not mop_name:
        mop_name = "UnknownMOP"

    # Avoid dimension leaks between different MOPs or M dimensions
    fair.clear_fair_cache()

    # 1. Resolve Sidecar Path
    if source_baseline:
        path = source_baseline
    else:
        # Sidecar resolution (proximity to class file)
        try:
            origin_file = inspect.getfile(mop.__class__)
            origin_dir = os.path.dirname(os.path.abspath(origin_file))
            path = os.path.join(origin_dir, f"{mop_name}_M{mop.M}.json")
        except:
            path = f"{mop_name}_M{mop.M}.json"

    # 2. Check Cache/File (Protocol: Loader)
    if os.path.exists(path) and not force:
        emit_output(
            f"moeabench: loading custom baselines for '{mop_name}' from sidecar.",
            markdown=f"Loading custom baselines for **{mop_name}** from sidecar."
        )
        base.register_baselines(path)
        return False

    # 3. Resolve Ground Truth (GT)
    gt = None
    
    # Protocol C: Empirical Search
    if source_search is not None:
        moea_name = source_search.__class__.__name__
        pop_size = getattr(source_search, 'population', 'unknown')
        gens = getattr(source_search, 'generations', 'unknown')
        
        effort_msg = ""
        if isinstance(pop_size, int) and isinstance(gens, int):
            effort_msg = f" (Estimated Effort: {pop_size * gens:,} evaluations)"
            
        emit_output(
            f"moeabench: discovering Ground Truth for '{mop_name}' via empirical search ({moea_name}){effort_msg}...",
            markdown=f"Discovering Ground Truth for **{mop_name}** via empirical search (`{moea_name}`){effort_msg}."
        )
        emit_output(
            "moeabench: warning: this exhaustive search may take from minutes to hours depending on your MOP's complexity.",
            markdown="> Warning: this exhaustive search may take from minutes to hours depending on your MOP's complexity."
        )

        from ..core.experiment import experiment
        exp = experiment(mop=mop, moea=source_search)
        # Ensure progress bar is captured if running in a terminal
        exp.run()
        # Use only non-dominated points as the truth
        gt = exp.non_dominated().objectives
        
    # Protocol B: Externally Supplied GT
    elif source_gt is not None:
        if isinstance(source_gt, str):
            emit_output(
                f"moeabench: loading Ground Truth for '{mop_name}' from CSV: {source_gt}",
                markdown=f"Loading Ground Truth for **{mop_name}** from CSV: `{source_gt}`"
            )
            gt = np.loadtxt(source_gt, delimiter=',')
        else:
            gt = np.asanyarray(source_gt)
            
    # Protocol A: Analytical (Default)
    else:
        emit_output(
            f"moeabench: generating analytical Ground Truth for '{mop_name}'...",
            markdown=f"Generating analytical Ground Truth for **{mop_name}**."
        )
        gt = mop.pf(n_points=2000)

    if gt is None or len(gt) == 0:
        raise ValueError(f"moeabench: Calibration failed. Problem '{mop_name}' returned empty Ground Truth.")

    # 4. Perform Statistical Calibration
    emit_output(
        "moeabench: calibrating noisy baselines (this may take a minute)...",
        markdown="Calibrating noisy baselines (this may take a minute)."
    )
    
    # Resolve K-Grid
    final_size = population_size or size
    if k_values:
        k_grid = k_values
    elif final_size:
        # Calibrate specifically for the requested size
        k_grid = [final_size]
    else:
        k_grid = K_GRID_DEFAULT
        
    # Only the native analytical protocol can certify independent ideal
    # samples here. A user-supplied/empirical GT may describe a different
    # manifold; never infer an analytical residue from an unrelated MOP.
    independent_ideal = (
        independent_analytical_reference(mop, gt)
        if source_gt is None and source_search is None else None
    )
    problem_data = _generate_baselines(mop_name, gt, k_grid, independent_ideal=independent_ideal)

    # 5. Save Sidecar
    import datetime
    import sys
    from ..system import version as lib_version
    
    sidecar_data = {
        "problem_id": mop_name,
        "mop_dimension": mop.M,
        "version": lib_version(show=False),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "timestamp": datetime.datetime.now().isoformat(),
        "schema": "baselines_v4_ecdf",
        "gt_reference": gt.tolist(),
        "problems": {
            mop_name: problem_data
        }
    }

    with open(path, "w") as f:
        json.dump(sidecar_data, f, indent=None)

    # 6. Register in current session
    base.register_baselines(sidecar_data)
    
    emit_output(
        f"moeabench: calibration complete. profile saved to: {path}",
        markdown=f"Calibration complete. Profile saved to: `{path}`"
    )
    return True

def _generate_baselines(name: str, gt: np.ndarray, k_grid: List[int],
                        independent_ideal: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Internal math engine for baseline calculation."""
    M = gt.shape[1]
    
    # 1. Normals and Clusters
    normals = _estimate_gt_normals(gt)
    c_cents, _ = base.get_ref_clusters(gt, c=32, seed=0)
    
    from ..progress import get_progress_bar
    
    problem_data = {}
    pbar = get_progress_bar(total=len(k_grid), desc=f"Calibration Profile: {name}")
    
    for i, k in enumerate(k_grid):
        k_str = str(k)
        
        # Scale and references
        s_fit = base.get_resolution_factor_k(gt, k, seed=0)
        u_ref = base.get_ref_uk(gt, k, seed=0)
        
        # Balance hist
        d_u = cdist(u_ref, c_cents)
        lab_u = np.argmin(d_u, axis=1)
        hist_ref = np.bincount(lab_u, minlength=len(c_cents)).astype(float)
        hist_ref /= np.sum(hist_ref)
        
        # CLOSENESS (Blur + Ideal Residue)
        sigma, u_blur = _calibrate_and_blur(gt, normals, s_fit, SEED_START + k)
        
        # ADR 0037: the finite-GT residue must use a distinct analytical
        # ideal-front sample, never an FPS subset *of the same GT*. For
        # empirical/externally supplied GT there may be no analytical oracle.
        # In that case the physical correction remains zero, explicitly
        # marked as unavailable, not misreported as a measured zero residue.
        if independent_ideal is None:
            ideal_res, ideal_res_method = 0.0, "unavailable_no_independent_ideal"
        else:
            ideal_res = estimate_ideal_residue(gt, independent_ideal, s_fit)
            ideal_res_method = "independent_analytical_pf"
        
        # Physical closeness subtracts the same ideal residue from every
        # sample. Its ECDF/anchor must be calibrated on the *corrected* values
        # too; comparing corrected observations against raw blurred baselines
        # would put the two sides of the Q-score on different scales.
        corrected_blur = np.maximum(0.0, u_blur - ideal_res)
        u_blur_ecdf = _downsample_ecdf(np.sort(corrected_blur))

        # Metrics storage
        k_data = {
            "closeness": {
                "uni50": 0.0,
                "rand50": float(np.median(corrected_blur)),
                "rand_ecdf": [float(x) for x in u_blur_ecdf],
                "ideal_res": ideal_res,
                "ideal_res_method": ideal_res_method
            }
        }
        
        # Other Metrics (Random and Ideal)
        rand_metrics = {m: [] for m in ["cov", "gap", "reg", "bal"]}
        ideal_metrics = {m: [] for m in ["cov", "gap", "reg", "bal"]}
        
        rng = np.random.RandomState(SEED_START + k)
        for _ in range(N_SAMPLES_ECDF):
            # Random Sample (Failure)
            pop_rand = rng.rand(k, M)
            m = _calc_all(pop_rand, gt, s_fit, u_ref, c_cents, hist_ref)
            for key in rand_metrics: rand_metrics[key].append(m[key])
            
        for _ in range(N_IDEAL_DRAWS):
            # Ideal Sample (Success)
            pop_uni = base.get_ref_uk(gt, k, seed=100 + _)
            m = _calc_all(pop_uni, gt, s_fit, u_ref, c_cents, hist_ref)
            for key in ideal_metrics: ideal_metrics[key].append(m[key])
            
        # Consolidate
        for m in ["cov", "gap", "reg", "bal"]:
            r_all = np.sort(rand_metrics[m])
            k_data[m] = {
                "uni50": float(np.median(ideal_metrics[m])),
                "rand50": float(np.median(r_all)),
                "rand_ecdf": [float(x) for x in r_all]
            }

        # ADR 0049: Q_HEADWAY has fixed 0/1 analytic anchors. Do not
        # substitute a single-population final distance or an undefined
        # HEADWAY statistic for a true paired P0/Pf trajectory.
        k_data["headway"] = {
            "ideal": 0.0,
            "no_progress": 1.0,
            "method": "analytic_anchors_adr0049",
            "status": "no_empirical_longitudinal_baseline",
        }
            
        problem_data[k_str] = k_data
        pbar.update_to(i + 1)
        
    pbar.close()
    return problem_data

def _calc_all(p, gt, s_fit, u_ref, c_cents, hist_ref):
    """Calculates all fair metrics for a sample."""
    return {
        # No HEADWAY: requires true P0/Pf from the same run (ADR 0049).
        "cov": fair.coverage(p, gt).value,
        "gap": fair.gap(p, gt).value,
        "reg": fair.regularity(p, u_ref).value,
        "bal": fair.balance(p, c_cents, hist_ref).value
    }

def _downsample_ecdf(sorted_vals: np.ndarray) -> np.ndarray:
    if len(sorted_vals) <= N_SAMPLES_ECDF:
        return sorted_vals
    indices = np.linspace(0, len(sorted_vals)-1, N_SAMPLES_ECDF).astype(int)
    return sorted_vals[indices]

def _calibrate_and_blur(gt, normals, s_fit, seed):
    """Calibrates sigma for Half-Normal blur."""
    target = 2.0
    sigma = 2.0 * s_fit
    
    # Iterative sigma search
    u_blur = None
    for it in range(6):
        rng = np.random.RandomState(seed + it)
        blurred = _generate_blur(gt, normals, sigma, rng)
        dists = cdist(blurred, gt)
        min_d = np.min(dists, axis=1)
        u_blur = min_d / s_fit
        med = np.median(u_blur)
        if abs(med - target) / target < 0.05:
            break
        sigma *= (target / med) if med > 1e-12 else 2.0
        
    return sigma, u_blur

def _generate_blur(gt, normals, sigma, rng):
    N, M = gt.shape
    blurred = np.zeros_like(gt)
    for i in range(N):
        n = normals[i]
        mag = abs(rng.normal(0, sigma))
        if np.linalg.norm(n) < 1e-12:
            n_rand = rng.normal(size=M)
            n_rand /= np.linalg.norm(n_rand)
            p_prime = gt[i] + mag * n_rand
        else:
            p_prime = gt[i] + mag * n
        # Boundary [0, inf)
        blurred[i] = np.maximum(p_prime, 0.0)
    return blurred

def _estimate_gt_normals(gt: np.ndarray) -> np.ndarray:
    N = len(gt)
    M = gt.shape[1]
    k_pca = int(np.clip(np.round(np.sqrt(N)), 10, 30))
    dists = cdist(gt, gt)
    indices = np.argsort(dists, axis=1)[:, 1:k_pca+1]
    
    rows = np.repeat(np.arange(N), k_pca)
    cols = indices.flatten()
    data = np.ones(N * k_pca)
    adj = csr_matrix((data, (rows, cols)), shape=(N, N))
    n_components, labels = connected_components(adj, directed=False)
    
    normals = np.zeros((N, M))
    for i in range(N):
        comp_mask = (labels == labels[i])
        comp_indices = np.where(comp_mask)[0]
        k_eff = min(k_pca, len(comp_indices) - 1)
        if k_eff < 2: continue
        
        d_comp = dists[i, comp_mask]
        idx_in_comp = np.argsort(d_comp)[1:k_eff+1]
        neighbors = gt[comp_indices[idx_in_comp]]
        
        mean = np.mean(neighbors, axis=0)
        centered = neighbors - mean
        cov = np.dot(centered.T, centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        n = eigvecs[:, 0]
        # Orient normal outward from origin (minimization logic)
        if np.dot(n, gt[i]) < 0:
            n = -n
        normals[i] = n / (np.linalg.norm(n) + 1e-12)
        
    return normals
