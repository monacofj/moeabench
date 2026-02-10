# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import os
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from .enums import DiagnosticStatus, DiagnosticProfile
from . import fair, qscore, baselines

# Thresholds for Q-Score (High-is-Better)
# Green >= 2/3, Yellow >= 1/3, Red < 1/3
THRESH_RESEARCH = 0.67
THRESH_INDUSTRY = 0.34
# Less strict profiles can be lower if needed, but 1/3 is the absolute floor.

@dataclass
class DiagnosticResult:
    """ Encapsulates the finding of an algorithmic audit. """
    status: DiagnosticStatus
    verdict: str
    metrics: Dict[str, Any]
    confidence: float
    description: str
    _rationale: str
    details: Dict[str, Any] = None 

    def rationale(self) -> str:
        return self._rationale

class PerformanceAuditor:
    """ Expert system for interpreting 5D Clinical Quality Scores. """
    
    @staticmethod
    def audit(metrics: Dict[str, float], 
              profile: DiagnosticProfile = DiagnosticProfile.EXPLORATORY) -> DiagnosticResult:
        
        # 1. Check Inputs
        input_ok = bool(metrics.get("input_ok", True))
        if not input_ok:
            return DiagnosticResult(
                status=DiagnosticStatus.UNDEFINED_INPUT,
                verdict="FAIL",
                metrics=metrics,
                confidence=0.0,
                description="Invalid input.",
                _rationale="Input validation failed."
            )
            
        baseline_ok = bool(metrics.get("baseline_ok", False))
        if not baseline_ok:
            return DiagnosticResult(
                status=DiagnosticStatus.UNDEFINED_BASELINE,
                verdict="FAIL", # Fail-Closed
                metrics=metrics,
                confidence=0.0,
                description="Missing baselines.",
                _rationale="Baselines unavailable for this MOP/K."
            )

        # 2. Extract Q-Scores
        # Default to 0.0 (Worst) if missing, to be safe
        q_fit = float(metrics.get("q_fit", 0.0))
        q_cov = float(metrics.get("q_coverage", 0.0))
        q_gap = float(metrics.get("q_gap", 0.0))
        q_reg = float(metrics.get("q_regularity", 0.0))
        q_bal = float(metrics.get("q_balance", 0.0))
        
        q_scores = {
            "FIT": q_fit,
            "COVERAGE": q_cov,
            "GAP": q_gap,
            "REGULARITY": q_reg,
            "BALANCE": q_bal
        }
        
        # 3. Determine Worst Dimension (Weakest Link)
        worst_dim = min(q_scores, key=q_scores.get)
        q_worst = q_scores[worst_dim]
        
        # 4. Determine Verdict
        # RESEARCH: Q >= 0.67
        # INDUSTRY: Q >= 0.34
        # FAIL: Q < 0.34
        
        verdict = "FAIL"
        status = DiagnosticStatus.UNDEFINED # Conceptual placeholder
        
        if q_worst >= THRESH_RESEARCH:
            # Passed strict research standard
            verdict = "PASS"
            status = DiagnosticStatus.IDEAL_FRONT
            rationale = "All dimensions meet Research Quality (>= 0.67)."
        elif q_worst >= THRESH_INDUSTRY:
            # Passed industry standard (loose)
            if profile == DiagnosticProfile.RESEARCH:
                verdict = "FAIL"
                rationale = f"Weakest link {worst_dim} ({q_worst:.2f}) below Research threshold."
            else:
                verdict = "PASS" # Industry/Exploratory pass
                rationale = f"Weakest link {worst_dim} ({q_worst:.2f}) meets Industry threshold."
            status = DiagnosticStatus.IDEAL_FRONT # Still "Ideal" topology, just lower quality? 
            # Or map specific issues based on dimension?
        else:
            # Absolute Fail
            verdict = "FAIL"
            rationale = f"Critically low quality in {worst_dim} ({q_worst:.2f})."
            
            # Map low score to Status
            if worst_dim == "FIT":
                status = DiagnosticStatus.SHIFTED_FRONT # Poor convergence
            elif worst_dim == "COVERAGE":
                status = DiagnosticStatus.COLLAPSED_FRONT # Poor coverage
            elif worst_dim == "GAP":
                status = DiagnosticStatus.GAPPED_COVERAGE
            elif worst_dim == "REGULARITY":
                status = DiagnosticStatus.NOISY_POPULATION # Irregular
            elif worst_dim == "BALANCE":
                status = DiagnosticStatus.BIASED_SPREAD
            else:
                status = DiagnosticStatus.SEARCH_FAILURE

        return DiagnosticResult(
            status=status,
            verdict=verdict,
            metrics=metrics,
            confidence=0.95,
            description=rationale,
            _rationale=f"[{profile.name}] {status.name} -> {verdict}. {rationale}"
        )

def audit(target: Any, 
          ground_truth: Optional[np.ndarray] = None,
          profile: DiagnosticProfile = DiagnosticProfile.EXPLORATORY) -> DiagnosticResult:
    """
    Main Entry Point.
    Computes FAIR metrics and Q-SCORES, then delegates to PerformanceAuditor.
    """
    metrics_data = {"input_ok": True, "baseline_ok": False}
    
    # 1. Extract Population (P) and Problem Info
    # Simplified extraction logic
    pop_objs = None
    mop_name = "Unknown"
    
    # Try to extract from dict or object
    if isinstance(target, dict):
        # Assume it's already metrics data? No, usually not passed here.
        # If passed pre-computed metrics, just delegate
        if "q_fit" in target:
            return PerformanceAuditor.audit(target, profile)
        return PerformanceAuditor.audit({"input_ok": False}, profile)
        
    if isinstance(target, np.ndarray):
        pop_objs = target
    elif hasattr(target, 'objectives'):
        pop_objs = target.objectives
    elif hasattr(target, 'last_pop'):
         pop_objs = target.last_pop.objectives
    
    if pop_objs is None:
         return PerformanceAuditor.audit({"input_ok": False}, profile)

    # 2. Extract Ground Truth (GT)
    # Needed for Fair Metrics
    if ground_truth is None:
        # Try to find from problem/experiment
        # TODO: Implement robust GT finding if not passed
        # For now, require ground_truth or fail
        return PerformanceAuditor.audit({"input_ok": False, "description": "GT required"}, profile)

    GT = ground_truth
    P = pop_objs
    
    # 3. K-Selection logic (Snap to Grid?)
    # Assume P is already the eval set? 
    # Or should we downsample?
    # New logic: The caller usually handles P.
    # But baselines are K-specific.
    K = len(P)
    
    # Snapshot MOP Name (needed for baselines)
    # Assuming the user ensures correct MOP name is active/known?
    # In 'target', if it's an Algorithm object, we can get it.
    if hasattr(target, 'problem') and hasattr(target.problem, 'name'):
        mop_name = target.problem.name
    elif hasattr(target, 'mop_name'):
        mop_name = target.mop_name
        
    metrics_data['mop_name'] = mop_name
    metrics_data['K'] = K
    metrics_data['K_raw'] = K # Assuming no downsampling happened here yet
    
    # 4. Compute Fair Metrics & Q-Scores
    try:
        # A. Baselines (for Ideal/Random logic inside qscore, but we need some context)
        # Verify baselines exist for this K/Mop (implicitly done inside qscore)
        pass # qscore will raise if missing
        
        # B. Shared Reference Objects (U_K, Hist)
        # Needed for Regularity/Balance FAIR calculation
        U_ref = baselines.get_ref_uk(GT, K, seed=0)
        centroids, _ = baselines.get_ref_clusters(GT, c=32, seed=0)
        
        # Reference Histogram for Balance
        d_u = baselines.cdist(U_ref, centroids)
        lab_u = np.argmin(d_u, axis=1)
        hist_ref = np.bincount(lab_u, minlength=len(centroids)).astype(float)
        hist_ref /= np.sum(hist_ref)
        
        # C. Compute FAIR Metrics (Physics)
        # s_gt = baselines.get_resolution_factor(GT) 
        s_k = baselines.get_resolution_factor_k(GT, K, seed=0)
        
        metrics_data['s_fit'] = s_k
        metrics_data['s_gt'] = baselines.get_resolution_factor(GT) # Keep for reference
        
        f_fit = fair.compute_fair_fit(P, GT, s_k)
        f_cov = fair.compute_fair_coverage(P, GT)
        f_gap = fair.compute_fair_gap(P, GT)
        f_reg = fair.compute_fair_regularity(P, U_ref)
        f_bal = fair.compute_fair_balance(P, centroids, hist_ref)
        
        metrics_data['fair_fit'] = f_fit
        metrics_data['fair_coverage'] = f_cov
        metrics_data['fair_gap'] = f_gap
        metrics_data['fair_regularity'] = f_reg
        metrics_data['fair_balance'] = f_bal
        
        # D. Compute Q-Scores (Engineering)
        q_fit = qscore.compute_q_fit(f_fit, mop_name, K)
        q_cov = qscore.compute_q_coverage(f_cov, mop_name, K)
        q_gap = qscore.compute_q_gap(f_gap, mop_name, K)
        q_reg = qscore.compute_q_regularity(f_reg, mop_name, K)
        q_bal = qscore.compute_q_balance(f_bal, mop_name, K)
        
        metrics_data['q_fit'] = q_fit
        metrics_data['q_coverage'] = q_cov
        metrics_data['q_gap'] = q_gap
        metrics_data['q_regularity'] = q_reg
        metrics_data['q_balance'] = q_bal
        
        metrics_data['baseline_ok'] = True
        
    except baselines.UndefinedBaselineError:
        metrics_data['baseline_ok'] = False
    except Exception as e:
        metrics_data['baseline_ok'] = False
        metrics_data['error'] = str(e)

    # 5. Verdict
    return PerformanceAuditor.audit(metrics_data, profile)
