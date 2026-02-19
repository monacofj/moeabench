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
        use_md = kwargs.get('markdown', True)
        if use_md:
            header = f"# Clinical Quality Audit: {self.mop_name} (K={self.k})"
            sep = ""
        else:
            header = f"\n=== CLINICAL QUALITY AUDIT: {self.mop_name} (K={self.k}) ==="
            sep = "-" * len(header.strip())
            
        lines = [header, sep] if not use_md else [header, ""]
        
        if use_md:
            table = ["| Dimension | Q-Score | Verdict |", "| :--- | :--- | :--- |"]
            for name, qres in self.scores.items():
                q = float(qres.value)
                matrix = qres._LABELS.get(qres.name, {})
                label = "Undefined"
                for thresh in sorted(matrix.keys(), reverse=True):
                    if q >= thresh:
                        label = matrix[thresh]
                        break
                table.append(f"| {name} | {q:.3f} | {label} |")
            lines.extend(table)
        else:
            # Terminal: List format matching FairAuditResult
            # Calculate max width for alignment
            width = max(len(name) for name in self.scores.keys()) if self.scores else 0
            
            for name, qres in self.scores.items():
                 # We call report with markdown=False and alignment width
                 lines.append(f"  {qres.report(markdown=False, width=width)}")
                
        return "\n".join(lines)

    def summary(self) -> str:
        """ Hierarchical clinical interpretation (The Decision Tree). """
        s = self.scores
        def q(n): return float(s[n].value) if n in s else 0.0

        # Gate 1: Proximity (Closeness)
        q_close = q("Q_CLOSENESS")
        if q_close < THRESH_INDUSTRY:
            msg = "Poor Convergence: The algorithm failed to approach the optimal front, resulting in a remote population profile."
            if q("Q_HEADWAY") >= THRESH_RESEARCH:
                msg = f"Poor Convergence: Despite effective progress in headway, the algorithm failed to approach the optimal manifold."
            return msg

        # Gate 2: Spatial Extent (Coverage & Gap)
        q_cov = q("Q_COVERAGE")
        q_gap = q("Q_GAP")
        
        # Worst Quartile check (0.25)
        if q_cov < 0.25 and q_gap < 0.25:
             return "Structural Failure: While some proximity was achieved, the front exhibits catastrophic collapse and fragmentation."

        # Level 3: Distribution Quality (Order)
        pathologies = []
        if q_cov < THRESH_INDUSTRY: pathologies.append("collapsed coverage")
        elif q_cov < THRESH_RESEARCH: pathologies.append("limited coverage")
        
        if q_gap < THRESH_INDUSTRY: pathologies.append("severe fragmentation")
        elif q_gap < THRESH_RESEARCH: pathologies.append("continuity breaches")
        
        if q("Q_REGULARITY") < THRESH_INDUSTRY: pathologies.append("unstructured spacing")
        elif q("Q_REGULARITY") < THRESH_RESEARCH: pathologies.append("irregular distribution")
        
        if q("Q_BALANCE") < THRESH_INDUSTRY: pathologies.append("skewed parity")
        elif q("Q_BALANCE") < THRESH_RESEARCH: pathologies.append("distribution bias")

        base = "Steady-state convergence confirmed." if q_close < THRESH_RESEARCH else "Asymptotic convergence confirmed."
        
        if not pathologies:
            return f"{base} Overall structural integrity meets certification standards."
            
        return f"{base} However, secondary structural flaws were detected: {', '.join(pathologies)}."

@dataclass
class FairAuditResult(Reportable):
    """ Results of a physical engineering audit. """
    metrics: Dict[str, fair.FairResult]
    
    def report(self, **kwargs) -> str:
        use_md = kwargs.get('markdown', True)
        header = "# Physical Engineering Audit" if use_md else "\n=== PHYSICAL ENGINEERING AUDIT ==="
        lines = [header, ""]
        for name, fres in self.metrics.items():
            if use_md:
                lines.append(f"- **{name}**: {float(fres.value):.4f} ({fres.description})")
            else:
                # Remove brackets and align colons (using 12 chars padding)
                lines.append(f"  {name:<12} : {float(fres.value):.4f} - {fres.description}")
        return "\n".join(lines)

@dataclass
class DiagnosticResult(Reportable):
    """ High-level synthesis of an algorithmic audit (The Biopsy). """
    q_audit_res: QualityAuditResult
    fair_audit_res: FairAuditResult
    status: DiagnosticStatus
    description: str

    def summary(self) -> str:
        """ Preferred fluid narrative for executive reporting. """
        if self.q_audit_res:
            return self.q_audit_res.summary()
        return self.description

    def report(self, **kwargs) -> str:
        use_md = kwargs.get('markdown', True)
        if use_md:
            header = "# MoeaBench Diagnostic Biopsy"
            status_line = f"**Primary Status**: {self.status.name.replace('_', ' ').title()}"
            exec_line = f"**Executive Summary**: {self.description}"
            sub_q = "## 1. Clinical Quality (Certification)"
            sub_f = "## 2. Physical Evidence (Facts)"
        else:
            header = "=== MOEABENCH DIAGNOSTIC BIOPSY ==="
            status_line = f"Primary Status: {self.status.name.replace('_', ' ').title()}"
            exec_line = f"Executive Summary: {self.description}"
            sub_q = ">> LAYER 1: CLINICAL QUALITY"
            sub_f = ">> LAYER 2: PHYSICAL EVIDENCE"

        lines = [
            header,
            status_line,
            exec_line,
            "",
            sub_q,
            self.q_audit_res.report(**kwargs) if self.q_audit_res else "N/A",
            "",
            sub_f,
            self.fair_audit_res.report(**kwargs) if self.fair_audit_res else "N/A"
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
                    "HEADWAY": DiagnosticStatus.SHIFTED_FRONT,
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
          profile: DiagnosticProfile = DiagnosticProfile.EXPLORATORY,
          **kwargs) -> DiagnosticResult:
    """
    [Cascade Entry Point]
    Computes FAIR metrics and Q-SCORES, then delegates to PerformanceAuditor for synthesis.
    """
    from .utils import _resolve_diagnostic_context
    
    # 1. Resolve Context (P, GT, s_k, Name, K)
    P, GT, s_k_mop, mop_name, K_raw = _resolve_diagnostic_context(target, ref=ground_truth, **kwargs)

    if P is None or GT is None:
         return PerformanceAuditor.audit_synthesis(None, None)

    # 2. K-Selection logic 
    K_target = baselines.snap_k(K_raw)
        
    # 4. Compute Fair Metrics & Q-Scores
    try:
        bases = baselines.load_offline_baselines()
        if "_gt_registry" in bases and mop_name in bases["_gt_registry"]:
             GT = np.array(bases["_gt_registry"][mop_name])

        # A. Shared Reference Objects
        U_ref = baselines.get_ref_uk(GT, K_target, seed=0)
        centroids, _ = baselines.get_ref_clusters(GT, c=32, seed=0)
        
        # Reference Histogram for Balance
        d_u = baselines.cdist(U_ref, centroids)
        lab_u = np.argmin(d_u, axis=1)
        hist_ref = np.bincount(lab_u, minlength=len(centroids)).astype(float)
        hist_ref /= np.sum(hist_ref)
        
        # B. Normalization Resolution
        # Use s_k from mop if available, otherwise calculate from GT
        s_k = s_k_mop if s_k_mop > 1e-12 else baselines.get_resolution_factor_k(GT, K_target, seed=0)
        
        # C. Compute FAIR Metrics (Physics)
        f_headway = fair.headway(P, GT, s_k)
        f_closeness_val = fair.closeness(P, GT, s_k) 
        f_cov = fair.coverage(P, GT)
        f_gap = fair.gap(P, GT)
        f_reg = fair.regularity(P, U_ref)
        f_bal = fair.balance(P, centroids, hist_ref)
        
        f_metrics = {
            "CLOSENESS": f_closeness_val, 
            "COVERAGE": f_cov,
            "GAP": f_gap,
            "REGULARITY": f_reg,
            "BALANCE": f_bal,
            "HEADWAY": f_headway
        }
        fair_res = PerformanceAuditor.audit_fair(f_metrics)
        
        # D. Compute Q-Scores (Engineering)
        q_h = qscore.q_headway(f_headway, problem=mop_name, k=K_target, s_k=s_k)
        q_clo = qscore.q_closeness(f_closeness_val, problem=mop_name, k=K_target)
        q_c = qscore.q_coverage(f_cov, problem=mop_name, k=K_target)
        q_g = qscore.q_gap(f_gap, problem=mop_name, k=K_target)
        q_r = qscore.q_regularity(f_reg, problem=mop_name, k=K_target)
        q_b = qscore.q_balance(f_bal, problem=mop_name, k=K_target)
        
        q_scores = {
            "Q_CLOSENESS": q_clo,
            "Q_COVERAGE": q_c,
            "Q_GAP": q_g,
            "Q_REGULARITY": q_r,
            "Q_BALANCE": q_b,
            "Q_HEADWAY": q_h
        }
        q_res = PerformanceAuditor.audit_quality(q_scores, mop=mop_name, k=K_target)
        
        # 5. Synthesis (The Biopsy)
        return PerformanceAuditor.audit_synthesis(q_res, fair_res)
        
    except baselines.UndefinedBaselineError:
        return PerformanceAuditor.audit_synthesis(None, None) 
    except Exception as e:
        raise e
