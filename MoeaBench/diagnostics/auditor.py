# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import os
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from .enums import DiagnosticStatus, DiagnosticProfile
from . import fair, qscore, baselines
from .base import Reportable

# Thresholds for Q-Score (High-is-Better)
# Green >= 2/3, Yellow >= 1/3, Red < 1/3
THRESH_RESEARCH = 0.67
THRESH_INDUSTRY = 0.34
# Less strict profiles can be lower if needed, but 1/3 is the absolute floor.

@dataclass
class QualityAuditResult(Reportable):
    """ Results of a clinical certification audit. """
    scores: Dict[str, qscore.QResult]
    mop_name: str
    k: int
    
    def report(self, **kwargs) -> str:
        lines = [f"# Clinical Quality Audit: {self.mop_name} (K={self.k})", ""]
        table = ["| Dimension | Q-Score | Verdict |", "| :--- | :--- | :--- |"]
        for name, qres in self.scores.items():
            q = float(qres.value)
            # Find label
            matrix = qres._LABELS.get(qres.name, {})
            label = "Undefined"
            for thresh in sorted(matrix.keys(), reverse=True):
                if q >= thresh:
                    label = matrix[thresh]
                    break
            table.append(f"| {name} | {q:.3f} | {label} |")
        lines.extend(table)
        return "\n".join(lines)

@dataclass
class FairAuditResult(Reportable):
    """ Results of a physical engineering audit. """
    metrics: Dict[str, fair.FairResult]
    
    def report(self, **kwargs) -> str:
        lines = ["# Physical Engineering Audit", ""]
        for name, fres in self.metrics.items():
            lines.append(f"- **{name}**: {float(fres.value):.4f} ({fres.description})")
        return "\n".join(lines)

@dataclass
class DiagnosticResult(Reportable):
    """ High-level synthesis of an algorithmic audit (The Biopsy). """
    q_audit_res: QualityAuditResult
    fair_audit_res: FairAuditResult
    status: DiagnosticStatus
    description: str

    def report(self, **kwargs) -> str:
        lines = [
            f"# MoeaBench Diagnostic Biopsy",
            f"**Primary Status**: {self.status.name.replace('_', ' ').title()}",
            f"**Executive Summary**: {self.description}",
            "",
            "## 1. Clinical Quality (Certification)",
            self.q_audit_res.report() if self.q_audit_res else "N/A",
            "",
            "## 2. Physical Evidence (Facts)",
            self.fair_audit_res.report() if self.fair_audit_res else "N/A"
        ]
        return "\n".join(lines)

class PerformanceAuditor:
    """ Expert system for interpreting Clinical Quality Scores. """
    
    @staticmethod
    def audit_fair(metrics: Dict[str, fair.FairResult]) -> FairAuditResult:
        """ Aggregates physical Fact results. """
        return FairAuditResult(metrics=metrics)
        
    @staticmethod
    def audit_quality(q_scores: Dict[str, qscore.QResult], 
                     mop: str = "Unknown", k: int = 0) -> QualityAuditResult:
        """ Aggregates Clinical Certification results. """
        return QualityAuditResult(scores=q_scores, mop_name=mop, k=k)

    @staticmethod
    def audit_synthesis(q_res: QualityAuditResult, 
                        f_res: FairAuditResult) -> DiagnosticResult:
        """ 
        The 'Biopsy' Logic. 
        Identifies pathologies without subjective weighting.
        """
        if q_res is None or f_res is None:
             return DiagnosticResult(None, None, DiagnosticStatus.UNDEFINED, "Audit failed or missing baselines.")

        # 1. Detect Pathologies (Q < 0.34)
        anomalies = []
        for name, qval in q_res.scores.items():
            if float(qval.value) < THRESH_INDUSTRY:
                anomalies.append(name.replace("Q_", "").title())
        
        status = DiagnosticStatus.IDEAL_FRONT
        desc = "Algorithm performance meets industry certification standards."
        
        if anomalies:
            status = DiagnosticStatus.SEARCH_FAILURE
            desc = f"Critical failures detected in: {', '.join(anomalies)}."
            
            # Simple heuristic mapping for single-mode failures
            if len(anomalies) == 1:
                mapping = {
                    "COVERAGE": DiagnosticStatus.COLLAPSED_FRONT,
                    "CLOSENESS": DiagnosticStatus.SHIFTED_FRONT,
                    "DENOISE": DiagnosticStatus.SHIFTED_FRONT,
                    "GAP": DiagnosticStatus.GAPPED_COVERAGE,
                    "BALANCE": DiagnosticStatus.BIASED_SPREAD,
                    "REGULARITY": DiagnosticStatus.NOISY_POPULATION
                }
                status = mapping.get(anomalies[0].upper(), status)

        # 2. Check Substandard range (0.34 <= Q < 0.67)
        elif any(float(q.value) < THRESH_RESEARCH for q in q_res.scores.values()):
            sub = [n.replace("Q_", "").title() for n, q in q_res.scores.items() if float(q.value) < THRESH_RESEARCH]
            desc = f"Performance is Substandard (Yellow Zone) in: {', '.join(sub)}."

        return DiagnosticResult(
            q_audit_res=q_res,
            fair_audit_res=f_res,
            status=status,
            description=desc
        )

def fair_audit(target: Any, ground_truth: Optional[np.ndarray] = None) -> FairAuditResult:
    """ Aggregates all physical (fair) metrics. """
    res = audit(target, ground_truth)
    return res.fair_audit_res

def q_audit(target: Any, ground_truth: Optional[np.ndarray] = None) -> QualityAuditResult:
    """ Aggregates all clinical (q) scores. """
    res = audit(target, ground_truth)
    return res.q_audit_res

def audit(target: Any, 
          ground_truth: Optional[np.ndarray] = None,
          profile: DiagnosticProfile = DiagnosticProfile.EXPLORATORY) -> DiagnosticResult:
    """
    [Cascade Entry Point]
    Computes FAIR metrics and Q-SCORES, then delegates to PerformanceAuditor for synthesis.
    """
    # 1. Extract Population (P) and Problem Info
    pop_objs = None
    mop_name = "Unknown"
    
    if isinstance(target, dict):
        # We don't support passing raw dicts here in the new architecture 
        # as we need to compute the fair/q cascade.
        return PerformanceAuditor.audit_synthesis(None, None) 
        
    if isinstance(target, np.ndarray):
        pop_objs = target
    elif hasattr(target, 'objectives'):
        pop_objs = target.objectives
    elif hasattr(target, 'last_pop'):
         pop_objs = target.last_pop.objectives
    
    if pop_objs is None:
         return PerformanceAuditor.audit_synthesis(None, None)

    # 2. Extract Ground Truth (GT)
    if ground_truth is None:
        return PerformanceAuditor.audit_synthesis(None, None)

    GT = ground_truth
    P = pop_objs
    
    # 3. K-Selection logic 
    K_raw = len(P)
    K_target = baselines.snap_k(K_raw)
    
    # Snapshot MOP Name
    if hasattr(target, 'problem') and hasattr(target.problem, 'name'):
        mop_name = target.problem.name
    elif hasattr(target, 'mop_name'):
        mop_name = target.mop_name
        
    # 4. Compute Fair Metrics & Q-Scores
    try:
        # A. Shared Reference Objects
        U_ref = baselines.get_ref_uk(GT, K_target, seed=0)
        centroids, _ = baselines.get_ref_clusters(GT, c=32, seed=0)
        
        # Reference Histogram for Balance
        d_u = baselines.cdist(U_ref, centroids)
        lab_u = np.argmin(d_u, axis=1)
        hist_ref = np.bincount(lab_u, minlength=len(centroids)).astype(float)
        hist_ref /= np.sum(hist_ref)
        
        # B. Normalization Resolution
        s_k = baselines.get_resolution_factor_k(GT, K_target, seed=0)
        
        # C. Compute FAIR Metrics (Physics)
        f_denoise = fair.fair_denoise(P, GT, s_k)
        f_closeness_raw = fair.fair_closeness(P, GT, s_k) 
        f_cov = fair.fair_coverage(P, GT)
        f_gap = fair.fair_gap(P, GT)
        f_reg = fair.fair_regularity(P, U_ref)
        f_bal = fair.fair_balance(P, centroids, hist_ref)
        
        f_metrics = {
            "DENOISE": fair.FairResult(value=f_denoise, name="DENOISE", description="Convergence precision"),
            "COVERAGE": fair.FairResult(value=f_cov, name="COVERAGE", description="Objective space coverage"),
            "GAP": fair.FairResult(value=f_gap, name="GAP", description="Population continuity"),
            "REGULARITY": fair.FairResult(value=f_reg, name="REGULARITY", description="Uniformity of spread"),
            "BALANCE": fair.FairResult(value=f_bal, name="BALANCE", description="Mass distribution")
        }
        fair_res = PerformanceAuditor.audit_fair(f_metrics)
        
        # D. Compute Q-Scores (Engineering)
        q_den = qscore.q_denoise(f_denoise, problem=mop_name, k=K_target)
        q_clo = qscore.q_closeness(f_closeness_raw, problem=mop_name, k=K_target)
        q_c = qscore.q_coverage(f_cov, problem=mop_name, k=K_target)
        q_g = qscore.q_gap(f_gap, problem=mop_name, k=K_target)
        q_r = qscore.q_regularity(f_reg, problem=mop_name, k=K_target)
        q_b = qscore.q_balance(f_bal, problem=mop_name, k=K_target)
        
        q_scores = {
            "Q_DENOISE": q_den,
            "Q_CLOSENESS": q_clo,
            "Q_COVERAGE": q_c,
            "Q_GAP": q_g,
            "Q_REGULARITY": q_r,
            "Q_BALANCE": q_b
        }
        q_res = PerformanceAuditor.audit_quality(q_scores, mop=mop_name, k=K_target)
        
        # 5. Synthesis (The Biopsy)
        return PerformanceAuditor.audit_synthesis(q_res, fair_res)
        
    except baselines.UndefinedBaselineError:
        return PerformanceAuditor.audit_synthesis(None, None) 
    except Exception as e:
        raise e
