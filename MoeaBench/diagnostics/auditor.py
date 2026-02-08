# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import json
import os
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from .enums import DiagnosticStatus, DiagnosticProfile

# Reference Data Configuration
REF_DATA_DIR = os.path.join(os.path.dirname(__file__), "resources/references")
K_GRID = [50, 100, 150, 200, 300]

# GT-only calibration (rand vs quasi-uniform) artifacts (per MOP).
GT_CALIBRATION_JSON = "calibration.json"
GT_CALIBRATION_PKG = "calibration_package.npz"

def _load_reference(mop_name):
    """Loads standardized reference package and baselines."""
    base_dir = os.path.join(REF_DATA_DIR, mop_name)
    if not os.path.exists(base_dir):
        return None, None
    
    pkg_path = os.path.join(base_dir, "ref_package.npz")
    json_path = os.path.join(base_dir, "baselines.json")
    
    try:
        pkg = np.load(pkg_path)
        with open(json_path, "r") as f:
            baselines = json.load(f)
        return pkg, baselines
    except:
        return None, None

def _load_gt_calibration(mop_name: str):
    """
    Loads GT-only calibration package for score-based auditing.

    Returns:
        (calib_json: dict|None, calib_pkg: np.load|None)
    """
    base_dir = os.path.join(REF_DATA_DIR, mop_name)
    if not os.path.exists(base_dir):
        return None, None

    json_path = os.path.join(base_dir, GT_CALIBRATION_JSON)
    pkg_path = os.path.join(base_dir, GT_CALIBRATION_PKG)
    if not (os.path.exists(json_path) and os.path.exists(pkg_path)):
        return None, None

    try:
        with open(json_path, "r") as f:
            calib = json.load(f)
        pkg = np.load(pkg_path)
        return calib, pkg
    except Exception:
        return None, None

def _snap_to_grid(n):
    """
    Snaps population size n to the nearest VALID K in our grid.
    STRICT MODE: Never choose a K greater than n (no upsampling).
    If n is smaller than the smallest grid point, we cannot use the grid.
    """
    valid_k = [k for k in K_GRID if k <= n]
    if not valid_k:
        return n # Fallback: absolute n (will miss cache but preserves density)
    return max(valid_k) # Closest downstream K

def _resample_to_K(pop, K):
    """
    Deterministic resampling/thinning to match cardinality K.
    STRICT MODE: Never upsample (duplicate). Only downsample or pass.
    """
    n = len(pop)
    if n == K: return pop
    if n < K:
        # Should be unreachable with new _snap_to_grid logic
        # But if it happens, we return RAW, never duplicate.
        return pop 
    else:
        # Downsample using unique snapping or just stride (simple for now)
        idx = np.linspace(0, n-1, K).astype(int)
        return pop[idx]

def _downsample_farthest_point(pts: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    Deterministic farthest-point downsampling (k-center style) to preserve coverage.
    This is used only when |ND| > K and we must choose the evaluated set P_K.
    """
    n = len(pts)
    if k >= n:
        return pts
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    chosen = [first]
    min_d = cdist(pts, pts[[first]]).reshape(-1)
    min_d[first] = 0.0
    while len(chosen) < k:
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        d_new = cdist(pts, pts[[nxt]]).reshape(-1)
        min_d = np.minimum(min_d, d_new)
        min_d[nxt] = 0.0
    return pts[np.array(chosen, dtype=int)]

def _nn_dists(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.array([0.0])
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)

def _score_from_calibration(val: float, uni_p50: float, rand_p50: float) -> float:
    """
    Score in [0,1]: 0 ~ quasi-uniform-at-K, 1 ~ typical random-at-K.
    Fail-closed: invalid calibration returns NaN and will trip baseline_ok=False.
    """
    denom = float(rand_p50) - float(uni_p50)
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    x = (float(val) - float(uni_p50)) / denom
    return float(np.clip(x, 0.0, 1.0))

@dataclass
class DiagnosticResult:
    """ Encapsulates the finding of an algorithmic audit. """
    status: DiagnosticStatus
    verdict: str
    verdicts: Dict[str, str]
    metrics: Dict[str, Any]
    confidence: float
    description: str
    _rationale: str
    details: Dict[str, Any] = None 

    def rationale(self) -> str:
        return self._rationale

class PerformanceAuditor:
    """ Expert system for interpreting multi-objective performance metrics. """
    
    @staticmethod
    def audit(metrics: Dict[str, float], 
              profile: DiagnosticProfile = DiagnosticProfile.EXPLORATORY,
              diameter: float = 1.0) -> DiagnosticResult:
        # Interpretable profile thresholds on score_global (0=best-possible-at-K, 1=typical-random-at-K).
        # These are "quartile-like" gates, not raw-distance magic numbers.
        SCORE_THRESH = {
            DiagnosticProfile.RESEARCH: 0.25,
            DiagnosticProfile.STANDARD: 0.375,
            DiagnosticProfile.INDUSTRY: 0.50,
            DiagnosticProfile.EXPLORATORY: 0.75,
        }

        input_ok = bool(metrics.get("input_ok", True))
        if not input_ok:
            verdicts = {p.name: "FAIL" for p in DiagnosticProfile}
            verdict = verdicts.get(profile.name, "FAIL")
            reason = str(metrics.get("input_error", "Invalid input for audit."))
            return DiagnosticResult(
                status=DiagnosticStatus.UNDEFINED_INPUT,
                verdict=verdict,
                verdicts=verdicts,
                metrics=metrics,
                confidence=0.0,
                description=reason,
                _rationale=f"[{profile.name}] UNDEFINED_INPUT -> {verdict}. {reason}",
            )

        baseline_ok = bool(metrics.get("baseline_ok", False))
        score_cov = float(metrics.get("score_cov", np.nan))
        score_pure = float(metrics.get("score_pure", np.nan))
        score_shape = float(metrics.get("score_shape", np.nan))
        score_global = float(metrics.get("score_global", np.nan))
        score_igd_mean = float(metrics.get("score_igd_mean", np.nan))
        score_igd_p95 = float(metrics.get("score_igd_p95", np.nan))

        # Status is diagnostic and profile-independent: "what happened with P?".
        # It is intentionally based on calibrated scores, not raw-distance cutoffs.
        if not baseline_ok or not np.isfinite(score_global):
            status = DiagnosticStatus.UNDEFINED_BASELINE
            rationale = "GT calibration unavailable or incompatible for this MOP/K."
        else:
            # IDEAL_FRONT: as good as the quasi-uniform-at-K reference (within the strictest tier).
            if max(score_cov, score_pure, score_shape) <= SCORE_THRESH[DiagnosticProfile.RESEARCH]:
                status = DiagnosticStatus.IDEAL_FRONT
                rationale = "All calibrated components are within the strict (research) gate."
            else:
                comps = {"cov": score_cov, "pure": score_pure, "shape": score_shape}
                dom = max(comps, key=lambda k: comps[k])
                if dom == "cov":
                    if score_shape > 0.50:
                        status = DiagnosticStatus.COLLAPSED_FRONT
                        rationale = "Coverage dominates and shape is very poor: collapse/degenerate support."
                    elif score_igd_p95 >= score_igd_mean + 0.05:
                        status = DiagnosticStatus.GAPPED_COVERAGE
                        rationale = "Coverage dominates with heavy-tail GT->P distances: gaps in coverage."
                    else:
                        status = DiagnosticStatus.SHIFTED_FRONT
                        rationale = "Coverage dominates without gap signature: systematic shift/under-coverage."
                elif dom == "pure":
                    status = DiagnosticStatus.NOISY_POPULATION
                    rationale = "Purity dominates: outliers/noise in P even if coverage is acceptable."
                else:
                    # Shape dominates: geometry can look fine but distribution is biased/non-uniform at K.
                    if score_cov <= 0.50 and score_pure <= 0.50:
                        status = DiagnosticStatus.BIASED_SPREAD
                        rationale = "Shape dominates with acceptable coverage/purity: biased/non-uniform spread."
                    else:
                        status = DiagnosticStatus.DISTORTED_COVERAGE
                        rationale = "Shape dominates with other issues: distorted distribution and coverage."

                if score_global >= 0.99:
                    status = DiagnosticStatus.SEARCH_FAILURE
                    rationale = "All calibrated components are as bad as typical random-at-K."

        verdicts = {}
        for p in DiagnosticProfile:
            thr = SCORE_THRESH[p]
            p_verdict = "FAIL"
            if baseline_ok and np.isfinite(score_global):
                if (score_global <= thr) and status not in {DiagnosticStatus.UNDEFINED_BASELINE, DiagnosticStatus.UNDEFINED_INPUT}:
                    p_verdict = "PASS"
            verdicts[p.name] = p_verdict

        verdict = verdicts.get(profile.name, "FAIL")
        metrics_str = f"score_global={score_global:.2f}, cov={score_cov:.2f}, pure={score_pure:.2f}, shape={score_shape:.2f}"
        full_rationale = f"[{profile.name}] {status.name} -> {verdict}. {rationale} ({metrics_str})"

        return DiagnosticResult(
            status=status, verdict=verdict, verdicts=verdicts,
            metrics=metrics, confidence=0.95, description=rationale, _rationale=full_rationale
        )

def audit(target: Any, 
          ground_truth: Optional[Any] = None,
          profile: DiagnosticProfile = DiagnosticProfile.EXPLORATORY) -> DiagnosticResult:
    """ Main auditing entry point. Uses standardized reference library. """
    metrics_data = {}
    diameter = 1.0
    
    def _get_diameter(pts: np.ndarray) -> float:
        if pts is None or len(pts) < 2: return 1.0
        diff = np.max(pts, axis=0) - np.min(pts, axis=0)
        return float(np.sqrt(np.sum(diff**2)))

    if isinstance(target, dict):
        metrics_data = target
    else:
        pop_objs, prob = None, None
        if hasattr(target, 'last_pop'):
            pop_objs = target.last_pop.objectives
            if hasattr(target, 'mop'): prob = target.mop
            elif hasattr(target, 'experiment') and target.experiment:
                prob = getattr(target.experiment, 'mop', None)
        elif isinstance(target, np.ndarray):
            pop_objs = target
            
        if pop_objs is not None:
            try:
                from MoeaBench.core.utils import is_non_dominated
                nd_mask = is_non_dominated(pop_objs)
                pop_objs = pop_objs[nd_mask]
            except Exception:
                try:
                    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
                    fronts = NonDominatedSorting().do(pop_objs)
                    if len(fronts) > 0:
                        pop_objs = pop_objs[fronts[0]]
                except Exception:
                    pass

            mop_name = prob.__class__.__name__ if prob else "Unknown"
            pkg, baselines = _load_reference(mop_name)
            
            pf_raw = ground_truth
            if pf_raw is None and prob is not None:
                if hasattr(prob, 'pf'): pf_raw = prob.pf()
                elif hasattr(prob, 'optimal_front'): pf_raw = prob.optimal_front()
            
            if pf_raw is not None:
                if pop_objs is None or pop_objs.ndim != 2:
                    return PerformanceAuditor.audit(
                        {"input_ok": False, "input_error": "Population objectives are missing or invalid."},
                        profile=profile,
                        diameter=diameter,
                    )
                if pop_objs.shape[1] != pf_raw.shape[1]:
                    return PerformanceAuditor.audit(
                        {
                            "input_ok": False,
                            "input_error": f"Dimension mismatch: pop M={pop_objs.shape[1]} vs GT M={pf_raw.shape[1]}",
                            "mop_name": mop_name,
                        },
                        profile=profile,
                        diameter=diameter,
                    )
                diameter = _get_diameter(pf_raw)
                n_raw = len(pop_objs)
                K = _snap_to_grid(n_raw)
                
                # Normalization
                calib, calib_pkg = _load_gt_calibration(mop_name)
                if calib_pkg is not None:
                    ideal, nadir = calib_pkg["ideal"], calib_pkg["nadir"]
                    pf_norm = calib_pkg["gt_norm"]
                elif pkg is not None:
                    ideal, nadir = pkg["ideal"], pkg["nadir"]
                    pf_norm = pkg["gt_norm"]
                else:
                    ideal, nadir = np.min(pf_raw, axis=0), np.max(pf_raw, axis=0)
                    denom = nadir - ideal
                    denom[denom == 0] = 1.0
                    pf_norm = (pf_raw - ideal) / denom
                
                denom = nadir - ideal
                denom[denom == 0] = 1.0
                pop_norm_full = (pop_objs - ideal) / denom

                # Strict: never upsample; choose evaluated P_K deterministically.
                # Use farthest-point downsample for better geometry preservation than stride.
                pop_norm = _downsample_farthest_point(pop_norm_full, K, seed=0)
                
                metrics_data['mop_name'] = mop_name
                metrics_data['n_raw'] = n_raw
                metrics_data['K'] = K
                metrics_data['M'] = pf_raw.shape[1]
                metrics_data['baseline_ok'] = False
                
                # Distances (normalized space): kept for report visualization.
                d_p_to = np.min(cdist(pop_norm, pf_norm), axis=1)
                d_gt_to = np.min(cdist(pf_norm, pop_norm), axis=1)
                metrics_data["min_dists"] = d_p_to.tolist()
                metrics_data["gt_min_dists"] = d_gt_to.tolist()

                # Robust / component metrics (normalized).
                igd_mean = float(np.mean(d_gt_to))
                igd_p95 = float(np.percentile(d_gt_to, 95))
                gd_p95 = float(np.percentile(d_p_to, 95))

                metrics_data["igd_mean"] = igd_mean
                metrics_data["igd_p95"] = igd_p95
                metrics_data["gd_p95"] = gd_p95

                # Score calibration (GT-only): rand vs quasi-uniform, same K.
                metrics_data["score_global"] = np.nan
                metrics_data["score_cov"] = np.nan
                metrics_data["score_pure"] = np.nan
                metrics_data["score_shape"] = np.nan
                metrics_data["score_igd_mean"] = np.nan
                metrics_data["score_igd_p95"] = np.nan

                if calib and calib_pkg is not None:
                    k_key = str(K)
                    k_meta = calib.get("metrics", {}).get(k_key)
                    if isinstance(k_meta, dict):
                        s = float(k_meta.get("s", np.nan))
                        metrics_data["s"] = s
                        pur = float(gd_p95 / s) if np.isfinite(s) and s > 1e-12 else float("nan")

                        u_key = f"u_ref_{K}"
                        if u_key in calib_pkg:
                            u_ref = calib_pkg[u_key]
                            nn_p = _nn_dists(pop_norm)
                            nn_u = _nn_dists(u_ref)
                            shape = float(wasserstein_distance(nn_p, nn_u))
                        else:
                            pur = float("nan")
                            shape = float("nan")

                        metrics_data["pur"] = pur
                        metrics_data["shape"] = shape

                        def _get_p50(metric_name: str):
                            mm = k_meta.get(metric_name, {})
                            return float(mm.get("uni_p50", np.nan)), float(mm.get("rand_p50", np.nan))

                        uni, rnd = _get_p50("IGD_mean")
                        score_igd_mean = _score_from_calibration(igd_mean, uni, rnd)
                        uni, rnd = _get_p50("IGD_95")
                        score_igd_p95 = _score_from_calibration(igd_p95, uni, rnd)
                        uni, rnd = _get_p50("PUR")
                        score_pure = _score_from_calibration(pur, uni, rnd)
                        uni, rnd = _get_p50("SHAPE")
                        score_shape = _score_from_calibration(shape, uni, rnd)

                        score_cov = float(np.nanmax([score_igd_mean, score_igd_p95]))
                        score_global = float(np.nanmax([score_cov, score_pure, score_shape]))

                        scores = [score_global, score_cov, score_pure, score_shape, score_igd_mean, score_igd_p95]
                        if all(np.isfinite(x) for x in scores):
                            metrics_data["baseline_ok"] = True
                            metrics_data["score_global"] = score_global
                            metrics_data["score_cov"] = score_cov
                            metrics_data["score_pure"] = score_pure
                            metrics_data["score_shape"] = score_shape
                            metrics_data["score_igd_mean"] = score_igd_mean
                            metrics_data["score_igd_p95"] = score_igd_p95
                            metrics_data["pur"] = pur
                            metrics_data["shape"] = shape
                
    return PerformanceAuditor.audit(metrics_data, profile=profile, diameter=diameter)
