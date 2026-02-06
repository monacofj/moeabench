# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import json
import os
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from .enums import DiagnosticStatus, DiagnosticProfile

# Reference Data Configuration
REF_DATA_DIR = os.path.join(os.path.dirname(__file__), "resources/references")
K_GRID = [50, 100, 200, 400, 800]

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
        
        MAX_EFF = {
            DiagnosticProfile.RESEARCH: 1.10,
            DiagnosticProfile.STANDARD: 1.60,
            DiagnosticProfile.INDUSTRY: 2.00,
            DiagnosticProfile.EXPLORATORY: 5.00
        }

        # 1. Super-Saturation Check (Warning Only)
        h_rel = metrics.get('h_rel', 0.0)
        is_saturated = h_rel > 1.05
        # We do NOT return early anymore. We just note it.

        # 2. Metric Extraction (Fail-Closed Defaults)
        # Default is infinity (worst possible), never 1.0
        igd_eff = metrics.get('igd_eff', np.inf)
        emd_eff_uniform = metrics.get('emd_eff_uniform', np.inf)
        gd_p95 = metrics.get('gd_p95', np.inf)
        
        # 3. Geometry Status (Physical Layer)
        # Convergence requires < 5x floor (stricter than 10x)
        GEOMETRY_EFF = 5.0
        
        # Normalized diameter is sqrt(M)
        norm_diameter = np.sqrt(metrics.get('M', 3))
        GEO_PURITY_THR_NORM = 0.05 * norm_diameter

        GEO_IGD = igd_eff <= GEOMETRY_EFF
        GEO_EMD = emd_eff_uniform <= GEOMETRY_EFF
        GEO_PURITY = gd_p95 <= GEO_PURITY_THR_NORM
        
        status = DiagnosticStatus.SEARCH_FAILURE
        rationale = ""
        
        if GEO_IGD: # Converged
            if GEO_PURITY:
                 if GEO_EMD:
                      # Stricter Ideal Front Requirements
                      if igd_eff < 1.3 and emd_eff_uniform < 1.5:
                          status = DiagnosticStatus.IDEAL_FRONT
                          rationale = "Ideal Front. Excellent convergence, purity, and topology."
                      else:
                          # It converged physically but isn't quite "Ideal"
                          status = DiagnosticStatus.NOISY_POPULATION
                          rationale = "Converged but efficiency sub-optimal (Noisy)."
                 else:
                      status = DiagnosticStatus.BIASED_SPREAD
                      rationale = "Biased Spread. Converged and pure, but distribution is skewed."
            else: # Bad Purity
                 if GEO_EMD:
                      status = DiagnosticStatus.NOISY_POPULATION
                      rationale = "Noisy Population. Right shape but fuzzy (Low Purity)."
                 else:
                      status = DiagnosticStatus.DISTORTED_COVERAGE
                      rationale = "Distorted Coverage. Roughly in place, but noisy and skewed."
        else: # Not Converged
             if GEO_EMD:
                  status = DiagnosticStatus.SHIFTED_FRONT
                  rationale = "Shifted Front. Good topology but displaced from GT."
             else:
                  if GEO_PURITY:
                       status = DiagnosticStatus.COLLAPSED_FRONT
                       rationale = "Collapsed Front. High purity but poor coverage."
                  else:
                       status = DiagnosticStatus.SEARCH_FAILURE
                       rationale = "Search Failure. No resemblance to the Pareto Front."

        # Override for Super-Saturation (if geometry is at least passable)
        if is_saturated and status not in [DiagnosticStatus.SEARCH_FAILURE, DiagnosticStatus.COLLAPSED_FRONT]:
             status = DiagnosticStatus.SUPER_SATURATION
             rationale += " (Super-Saturation Detected)"

        # 4. Determine Verdicts
        verdicts = {}
        for p in DiagnosticProfile:
            p_val = p.value / 100.0
            p_cert_gd = gd_p95 <= (p_val * norm_diameter)
            
            p_eff = MAX_EFF.get(p, 5.0)
            p_cert_igd = igd_eff <= p_eff
            
            # CRITICAL FIX: EMD check is mandatory for ALL profiles now
            # Standard/Industry can tolerate more (e.g., 2.0x), but can't be infinite.
            p_cert_emd = emd_eff_uniform <= p_eff
            
            p_verdict = "FAIL"
            
            # Fail-Closed: If any metric is infinite, verdict is automatic FAIL
            if np.isinf(igd_eff) or np.isinf(emd_eff_uniform):
                p_verdict = "FAIL"
            else:
                if p == DiagnosticProfile.RESEARCH:
                    if status in [DiagnosticStatus.IDEAL_FRONT, DiagnosticStatus.SUPER_SATURATION] and p_cert_igd and p_cert_emd and p_cert_gd:
                         p_verdict = "PASS"
                else: 
                     # For others, we accept wider status but require metrics to pass limits
                     if p_cert_igd and p_cert_emd and status not in [DiagnosticStatus.SEARCH_FAILURE, DiagnosticStatus.COLLAPSED_FRONT]:
                         p_verdict = "PASS"
            
            verdicts[p.name] = p_verdict

        verdict = verdicts.get(profile.name, "FAIL")
        metrics_str = f"IGD_eff={igd_eff:.2f}x, EMD_eff={emd_eff_uniform:.2f}x"
        full_rationale = f"[{profile.name}] {status.name} -> {verdict}. {rationale} ({metrics_str})"

        # Persist efficiencies for reporting
        metrics['igd_eff'] = igd_eff
        metrics['emd_eff_uniform'] = emd_eff_uniform
        metrics['gd_p95_norm'] = gd_p95

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
            mop_name = prob.__class__.__name__ if prob else "Unknown"
            pkg, baselines = _load_reference(mop_name)
            
            pf_raw = ground_truth
            if pf_raw is None and prob is not None:
                if hasattr(prob, 'pf'): pf_raw = prob.pf()
                elif hasattr(prob, 'optimal_front'): pf_raw = prob.optimal_front()
            
            if pf_raw is not None:
                diameter = _get_diameter(pf_raw)
                n_raw = len(pop_objs)
                K = _snap_to_grid(n_raw)
                
                # Strict: _resample_to_K will no longer upsample
                pop_K = _resample_to_K(pop_objs, K)
                
                # Normalization
                if pkg is not None: 
                    ideal, nadir = pkg['ideal'], pkg['nadir']
                    pf_norm = pkg['gt_norm']
                else: 
                    ideal, nadir = np.min(pf_raw, axis=0), np.max(pf_raw, axis=0)
                    denom = nadir - ideal
                    denom[denom == 0] = 1.0
                    pf_norm = (pf_raw - ideal) / denom
                
                denom = nadir - ideal
                denom[denom == 0] = 1.0
                pop_norm = (pop_K - ideal) / denom
                
                metrics_data['mop_name'] = mop_name
                metrics_data['n_raw'] = n_raw
                metrics_data['K'] = K
                metrics_data['M'] = pf_raw.shape[1]
                
                # 1. Purity (Normalized Space)
                dist_matrix = cdist(pop_norm, pf_norm)
                min_dists = np.min(dist_matrix, axis=1)
                metrics_data['gd_p95'] = float(np.percentile(min_dists, 95))
                metrics_data['min_dists'] = min_dists.tolist() 
                
                # 2. Coverage (Normalized Space)
                gt_min_dists = np.min(dist_matrix, axis=0)
                metrics_data['gt_min_dists'] = gt_min_dists.tolist() 
                
                # 3. Core Metrics (Normalized)
                igd_raw = float(np.mean(gt_min_dists)) 
                
                from scipy.stats import wasserstein_distance
                def _emd_avg(pts, ref_pts):
                    M = pts.shape[1]
                    return np.mean([wasserstein_distance(pts[:, m], ref_pts[:, m]) for m in range(M)])
                
                if pkg is not None and f"uni_{K}" in pkg:
                    emd_uni_raw = _emd_avg(pop_norm, pkg[f"uni_{K}"])
                else:
                    emd_uni_raw = _emd_avg(pop_norm, pf_norm)
                
                # 3. Efficiency Mapping (Scientific Rigor: Fail-Closed)
                metrics_data['igd_eff'] = np.inf 
                metrics_data['emd_eff_uniform'] = np.inf

                if baselines and str(K) in baselines['K_data']:
                    bk = baselines['K_data'][str(K)]
                    # Check for valid floors
                    igd_f = bk.get('igd_floor', 0.0)
                    emd_f = bk.get('emd_uni_floor', 0.0)
                    
                    if igd_f > 1e-12:
                        metrics_data['igd_eff'] = igd_raw / igd_f
                    if emd_f > 1e-12:
                        metrics_data['emd_eff_uniform'] = emd_uni_raw / emd_f
                else:
                    # Missing baseline -> inf (FAIL), never 1.0
                    pass 
                
                metrics_data['igd'], metrics_data['emd'] = igd_raw, emd_uni_raw
                
    return PerformanceAuditor.audit(metrics_data, profile=profile, diameter=diameter)
