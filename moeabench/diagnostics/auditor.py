# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import os
from typing import Optional, Any, Dict, List, Union
from dataclasses import dataclass
from .enums import DiagnosticStatus
from . import fair, qscore, baselines, calibration
from .base import Reportable

# Thresholds for Q-Score (High-is-Better)
# Green >= 2/3, Yellow >= 1/3, Red < 1/3
THRESH_RESEARCH = 0.67
THRESH_INDUSTRY = 0.34
# Less strict profiles can be lower if needed, but 1/3 is the absolute floor.

_DISPLAY_NAMES = {
    "CLOSENESS": "Closeness",
    "COVERAGE": "Coverage",
    "GAP": "Gap",
    "REGULARITY": "Regularity",
    "BALANCE": "Balance",
    "HEADWAY": "Headway",
    "Q_CLOSENESS": "Closeness",
    "Q_COVERAGE": "Coverage",
    "Q_GAP": "Gap",
    "Q_REGULARITY": "Regularity",
    "Q_BALANCE": "Balance",
    "Q_HEADWAY": "Headway",
}

def _display_name(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _fair_description(name: str, description: str) -> str:
    replacements = {
        "HEADWAY": "Fraction of the initial search error left unreduced.",
        "COVERAGE": "Average distance from target manifold to nearest solution.",
        "GAP": "Largest hole detected on the manifold.",
        "REGULARITY": "Deviation from ideal lattice spacing (Wasserstein distance).",
        "BALANCE": "Distribution bias across regions (JS Divergence).",
    }
    return replacements.get(name, description)

def _quality_executive_summary(scores: Dict[str, qscore.QResult]) -> str:
    """Hierarchical clinical interpretation (decision tree)."""
    def q(name: str) -> float:
        return float(scores[name].value) if name in scores else 0.0

    # Gate 1: Proximity (Closeness)
    q_close = q("Q_CLOSENESS")
    if q_close < THRESH_INDUSTRY:
        msg = "Poor convergence: the final population remained far from the target manifold."
        if q("Q_HEADWAY") >= THRESH_RESEARCH:
            msg = "Poor convergence: despite strong progress, the final population remained displaced from the target manifold."
        return msg

    # Gate 2: Spatial Extent (Coverage & Gap)
    q_cov = q("Q_COVERAGE")
    q_gap = q("Q_GAP")
    if q_cov < 0.25 and q_gap < 0.25:
        return "Structural failure: proximity was achieved, but the front is both collapsed and fragmented."

    # Gate 3: Distribution Quality (Order)
    pathologies = []
    if q_cov < THRESH_INDUSTRY: pathologies.append("collapsed coverage")
    elif q_cov < THRESH_RESEARCH: pathologies.append("limited coverage")

    if q_gap < THRESH_INDUSTRY: pathologies.append("severe fragmentation")
    elif q_gap < THRESH_RESEARCH: pathologies.append("continuity breaches")

    if q("Q_REGULARITY") < THRESH_INDUSTRY: pathologies.append("unstructured spacing")
    elif q("Q_REGULARITY") < THRESH_RESEARCH: pathologies.append("irregular distribution")

    if q("Q_BALANCE") < THRESH_INDUSTRY: pathologies.append("skewed parity")
    elif q("Q_BALANCE") < THRESH_RESEARCH: pathologies.append("distribution bias")

    base = "Convergence is adequate." if q_close < THRESH_RESEARCH else "Convergence is strong."
    if not pathologies:
        return f"{base} Performance is strong across all evaluated dimensions."
    return f"{base} Secondary weaknesses were detected in: {', '.join(pathologies)}."

@dataclass
class QualityAuditResult(Reportable):
    """ Results of a clinical quality validation. """
    scores: Dict[str, qscore.QResult]
    mop_name: str
    k: int
    experiment_name: Optional[str] = None
    
    @property
    def verdicts(self) -> Dict[str, str]:
        """Returns a dictionary of human-readable labels for each Q-Score."""
        return {name: qres.verdict for name, qres in self.scores.items()}
    
    def report(self, show: bool = True, full: bool = False, **kwargs) -> str:
        if not full:
            return self._render_report(_quality_executive_summary(self.scores), show, **kwargs)

        use_md = kwargs.get('markdown', self._is_notebook())
        subject = self.experiment_name or self.mop_name
        if use_md:
            header = "# Clinical Quality Scores"
            sep = ""
        else:
            header = "Clinical Quality Scores"
            sep = ""
            
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
                table.append(f"| {_display_name(name)} | {q:.3f} | {label} |")
            lines.extend(table)
        else:
            # Terminal: List format matching FairAuditResult
            # Calculate max width for alignment
            width = max(len(_display_name(name)) for name in self.scores.keys()) if self.scores else 0
            
            for name, qres in self.scores.items():
                 lines.append(f"  {qres.report(show=False, markdown=False, width=width)}")
                
        content = "\n".join(lines)
        return self._render_report(content, show, **kwargs)

@dataclass
class FairAuditResult(Reportable):
    """ Results of FAIR performance metrics. """
    metrics: Dict[str, fair.FairResult]
    
    def report(self, show: bool = True, full: bool = False, **kwargs) -> str:
        use_md = kwargs.get('markdown', self._is_notebook())
        header = "# FAIR Performance Metrics" if use_md else "FAIR Performance Metrics"
        lines = [header, ""]
        for name, fres in self.metrics.items():
            disp = _display_name(name)
            desc = _fair_description(name, fres.description)
            if use_md:
                lines.append(f"- **{disp}**: {float(fres.value):.4f} ({desc})")
            else:
                lines.append(f"  {disp:<12} : {float(fres.value):.4f} - {desc}")
        content = "\n".join(lines)
        return self._render_report(content, show, **kwargs)

# Compatibility Alias
FairAuditResult = FairAuditResult


@dataclass
class DiagnosticResult(Reportable):
    """ High-level synthesis of an algorithmic audit. """
    q_audit_res: QualityAuditResult
    fair_audit_res: FairAuditResult
    status: DiagnosticStatus
    description: str = ""
    reproducibility: Optional[Dict[str, Any]] = None,
    diagnostic_context: Optional[Dict[str, Any]] = None
    experiment_name: Optional[str] = None

    @property
    def quality(self) -> QualityAuditResult:
        """Access the Quality (Q-Score) audit results."""
        return self.q_audit_res

    @property
    def fr(self) -> FairAuditResult:
        """Access the FR (Physical) audit results."""
        return self.fair_audit_res

    # Compatibility Alias
    @property
    def fair(self) -> FairAuditResult:
        """Alias for fr to maintain backward compatibility."""
        return self.fair_audit_res


    @property
    def verdicts(self) -> Dict[str, str]:
        """Proxy to access Q-Score verdicts directly."""
        return self.q_audit_res.verdicts

    def report(self, show: bool = True, full: bool = False, **kwargs) -> str:
        if not full:
            if self.status == DiagnosticStatus.MISSING_BASELINE:
                return self._render_report(self.description, show, **kwargs)
            if self.q_audit_res:
                return self._render_report(_quality_executive_summary(self.q_audit_res.scores), show, **kwargs)
            return self._render_report(self.description, show, **kwargs)

        use_md = kwargs.get('markdown', self._is_notebook())
        subject = self.experiment_name or (
            self.q_audit_res.experiment_name if self.q_audit_res else None
        ) or (
            self.q_audit_res.mop_name if self.q_audit_res else None
        ) or "Unknown"
        if use_md:
            header = f"# MoeaBench Clinical Report: {subject}"
            status_line = f"**Primary Status**: {self.status.name.replace('_', ' ').title()}"
            exec_line = f"**Executive Summary**: {self.description}"
            sub_f = "## FAIR Performance Metrics"
            sub_q = "## Clinical Quality Scores"
        else:
            header = f"Clinical Report: {subject}"
            status_line = f"Primary Status: {self.status.name.replace('_', ' ').title()}"
            exec_line = f"Executive Summary: {self.description}"
            sub_f = "FAIR Performance Metrics"
            sub_q = "Clinical Quality Scores"

        # We call nested reports with show=False to gather their strings
        if self.status == DiagnosticStatus.MISSING_BASELINE:
             q_rep = self.description
        else:
             q_rep = self.q_audit_res.report(show=False, full=True, **kwargs) if self.q_audit_res else "N/A"
        f_rep = self.fair_audit_res.report(show=False, full=True, **kwargs) if self.fair_audit_res else "N/A"

        lines = [
            status_line,
            exec_line,
            "",
            f_rep,
            "",
            q_rep
        ]
        if header:
            lines.insert(0, header)

        if self.reproducibility:
            sub_r = "## Reproducibility Metadata" if use_md else "Reproducibility Metadata"
            r_info = [f"- **{k.replace('_', ' ').title()}**: {v}" for k, v in self.reproducibility.items()] if use_md else \
                     [f"{k.replace('_', ' ').title()}: {v}" for k, v in self.reproducibility.items()]
            lines += ["", sub_r] + r_info

        content = "\n".join(lines)
        return self._render_report(content, show, **kwargs)

class PerformanceAuditor:
    """ Expert system for interpreting Clinical Quality Scores. """

    _SINGLE_FAILURE_SUMMARIES = {
        DiagnosticStatus.COLLAPSED_FRONT:
            "The population approached the manifold but did not cover it with sufficient extent.",
        DiagnosticStatus.SHIFTED_FRONT:
            "The final population remained displaced from the target manifold.",
        DiagnosticStatus.GAPPED_COVERAGE:
            "The population spans the manifold broadly, but significant holes remain along the front.",
        DiagnosticStatus.BIASED_SPREAD:
            "The population reached the manifold, but its distribution remains uneven across regions.",
        DiagnosticStatus.IRREGULAR_FRONT:
            "The population is close to the target and broadly well distributed, but local spacing remains irregular.",
        DiagnosticStatus.SEARCH_FAILURE:
            "A single quality dimension received a very low score.",
    }
    
    @staticmethod
    def audit_fr(metrics: Dict[str, fair.FairResult]) -> FairAuditResult:
        """ Aggregates physical Fact results. """
        return FairAuditResult(metrics=metrics)

    @staticmethod
    def calibrate(mop: Any, population_size: Optional[int] = None, 
                 size: Optional[int] = None, **kwargs) -> bool:
        """ 
        [mb.diagnostics.PerformanceAuditor.calibrate]
        Generates Clinical Baselines for a MOP with a specific population size.
        """
        return calibration.calibrate_mop(mop, population_size=population_size, size=size, **kwargs)

    # Compatibility Alias
    audit_fair = audit_fr

        
    @staticmethod
    def audit_quality(q_scores: Dict[str, qscore.QResult], 
                     mop: str = "Unknown", k: int = 0,
                     experiment_name: Optional[str] = None) -> QualityAuditResult:
        """ Aggregates Clinical Quality results. """
        return QualityAuditResult(
            scores=q_scores,
            mop_name=mop,
            k=k,
            experiment_name=experiment_name,
        )

    @staticmethod
    def audit_synthesis(q_res: Optional[QualityAuditResult], 
                         f_res: Optional[FairAuditResult],
                         status: DiagnosticStatus = DiagnosticStatus.UNDEFINED,
                         description: str = "Audit failed.",
                         reproducibility: Optional[Dict[str, Any]] = None,
                         experiment_name: Optional[str] = None) -> DiagnosticResult:
        """ 
        The 'Synthesis' Logic. 
        Identifies pathologies without subjective weighting.
        """
        if q_res is None or f_res is None:
             if status == DiagnosticStatus.MISSING_BASELINE:
                 return DiagnosticResult(q_audit_res=None, fair_audit_res=None, status=status, description=description, reproducibility=reproducibility, experiment_name=experiment_name)
             return DiagnosticResult(
                 q_audit_res=None,
                 fair_audit_res=None,
                 status=DiagnosticStatus.UNDEFINED,
                 description="The audit could not determine a reliable clinical classification for this result.",
                 reproducibility=reproducibility,
                 experiment_name=experiment_name,
             )

        # 1. Detect Pathologies (Q < 0.34)
        anomalies = []
        for name, qval in q_res.scores.items():
            if float(qval.value) < THRESH_INDUSTRY:
                anomalies.append(name.replace("Q_", "").title())
        
        status = DiagnosticStatus.IDEAL_FRONT
        desc = "Performance is strong across all evaluated dimensions."
        
        if anomalies:
            status = DiagnosticStatus.SEARCH_FAILURE
            desc = f"Multiple quality dimensions received very low scores: {', '.join(anomalies)}."
            
            # Simple heuristic mapping for single-mode failures
            if len(anomalies) == 1:
                mapping = {
                    "COVERAGE": DiagnosticStatus.COLLAPSED_FRONT,
                    "CLOSENESS": DiagnosticStatus.SHIFTED_FRONT,
                    "HEADWAY": DiagnosticStatus.SHIFTED_FRONT,
                    "GAP": DiagnosticStatus.GAPPED_COVERAGE,
                    "BALANCE": DiagnosticStatus.BIASED_SPREAD,
                    "REGULARITY": DiagnosticStatus.IRREGULAR_FRONT
                }
                status = mapping.get(anomalies[0].upper(), status)
                desc = PerformanceAuditor._SINGLE_FAILURE_SUMMARIES.get(
                    status,
                    f"{anomalies[0]} received a very low score."
                )

        # 2. Check Substandard range (0.34 <= Q < 0.67)
        elif any(float(q.value) < THRESH_RESEARCH for q in q_res.scores.values()):
            sub = [n.replace("Q_", "").title() for n, q in q_res.scores.items() if float(q.value) < THRESH_RESEARCH]
            desc = f"Some quality dimensions remain below the target range: {', '.join(sub)}."

        return DiagnosticResult(
            q_audit_res=q_res,
            fair_audit_res=f_res,
            status=status,
            description=desc,
            reproducibility=reproducibility,
            experiment_name=experiment_name or q_res.experiment_name
        )

def fair_audit(target: Any, ground_truth: Optional[np.ndarray] = None) -> FairAuditResult:
    """ Aggregates all physical (fr) metrics. """
    res = audit(target, ground_truth)
    return res.fair_audit_res

# Compatibility Alias
fair_audit = fair_audit

def q_audit(target: Any, ground_truth: Optional[np.ndarray] = None) -> QualityAuditResult:
    """ Aggregates all clinical (q) scores. """
    res = audit(target, ground_truth)
    return res.q_audit_res

def audit(target: Any, 
          ground_truth: Optional[np.ndarray] = None,
          source_baseline: Optional[Union[str, Dict[str, Any]]] = None,
          quality: bool = True,
          **kwargs) -> DiagnosticResult:
    """
    [Cascade Entry Point]
    Computes FAIR metrics and, optionally, Q-SCORES, then delegates to synthesis.
    """
    import os
    import inspect
    from .utils import _resolve_diagnostic_context
    from ..system import info
    
    # 0. Resolve Context (P, GT, s_k, Name, K)
    ctx = _resolve_diagnostic_context(target, ref=ground_truth, **kwargs)
    P = ctx['P_final']
    GT = ctx['GT']
    s_k_mop = ctx['s_k']
    mop_name = ctx['problem']
    K_raw = ctx['k']
    experiment_name = getattr(target, 'name', None) or getattr(getattr(target, 'source', None), 'name', None)
    
    if P is None or GT is None:
         return PerformanceAuditor.audit_synthesis(None, None, experiment_name=experiment_name)

    # 1. Capture Reproducibility Metadata (Dimension-Aware)
    r_info = info(show=False)
    target_m = P.shape[1] if P is not None else None
    
    # 1.1 Smart Sidecar Discovery (Dimension-Aware)
    if not source_baseline and P is not None:
        mop = getattr(target, 'mop', None) or getattr(getattr(target, 'source', None), 'mop', None)
        if mop and hasattr(mop, 'M'):
            try:
                origin_file = inspect.getfile(mop.__class__)
                origin_dir = os.path.dirname(os.path.abspath(origin_file))
                sidecar_m = os.path.join(origin_dir, f"{mop_name}_M{mop.M}.json")
                if os.path.exists(sidecar_m):
                    source_baseline = sidecar_m
            except:
                pass

    # 1.2 Clear Caches
    fair.clear_fair_cache()

    # 1.3 Context Manager for External Baselines
    from contextlib import nullcontext
    cm = baselines.use_baselines(source_baseline) if source_baseline else nullcontext()
    
    with cm:
        # 1.4 Append Baseline DNA (after CM to capture sidecar info)
        try:
            bases = baselines.load_offline_baselines(target_m=target_m)
            r_info["baseline_version"] = bases.get("version", "Unknown")
            r_info["baseline_schema"] = bases.get("schema", "Legacy")
        except:
            r_info["baseline_version"] = "None"

        # 2. K-Selection logic (Standardized snap grid)
        # Use MOP name if provided to avoid snapping errors in localMaOP cases
        # LEGACY/REPRODUCIBILITY: Only enable smart snapping for MaOP (M > 3) 
        # to ensure standard benchmarks stick to the canonical [50, 100, 150, 200] grid.
        K_target = baselines.snap_k(K_raw, problem=mop_name if (target_m or 0) > 3 else None)
            
        # 4. Compute Fair Metrics & Q-Scores
        try:
            bases = baselines.load_offline_baselines(target_m=target_m)
            
            # Smart GT resolution: allow registry to provide the canonical 10k-point reference
            # if the dimension matches. This is necessary for numerical consistency across versions.
            if "_gt_registry" in bases:
                 gt_registry = bases["_gt_registry"]
                 gt_key = f"{mop_name}__M{P.shape[1]}"
                 GT_raw = None
                 if gt_key in gt_registry:
                     GT_raw = np.array(gt_registry[gt_key])
                 elif mop_name in gt_registry:
                     GT_raw = np.array(gt_registry[mop_name])
                 if GT_raw is not None:
                     if GT_raw.shape[1] == P.shape[1]:
                         GT = GT_raw
                     else:
                         # Force baseline missing if dimensions don't match
                         raise baselines.UndefinedBaselineError(
                             f"Dimension mismatch: P={P.shape[1]}, GT_registry={GT_raw.shape[1]}"
                         )

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
            
            # C. Compute FR Metrics (Physics)
            f_headway = fair.headway(P, GT, s_k, problem=mop_name, k=K_target, initial_data=ctx.get('P_initial'))
            f_closeness_val = fair.closeness(P, GT, s_k, problem=mop_name, k=K_target) 
            f_cov = fair.coverage(P, GT, problem=mop_name, k=K_target)
            f_gap = fair.gap(P, GT, problem=mop_name, k=K_target)
            f_reg = fair.regularity(P, U_ref, problem=mop_name, k=K_target)
            f_bal = fair.balance(P, centroids, hist_ref, problem=mop_name, k=K_target)
            
            f_metrics = {
                "CLOSENESS": f_closeness_val, 
                "COVERAGE": f_cov,
                "GAP": f_gap,
                "REGULARITY": f_reg,
                "BALANCE": f_bal,
                "HEADWAY": f_headway
            }
            fr_res = PerformanceAuditor.audit_fr(f_metrics)

            # Optional quality stage: return FAIR-only diagnostics when disabled.
            if not quality:
                return DiagnosticResult(
                    q_audit_res=None,
                    fair_audit_res=fr_res,
                    status=DiagnosticStatus.UNDEFINED,
                    description="FAIR metrics were computed, but clinical scoring was not requested.",
                    reproducibility=r_info,
                    diagnostic_context=ctx,
                    experiment_name=experiment_name
                )
            
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
            q_res = PerformanceAuditor.audit_quality(
                q_scores,
                mop=mop_name,
                k=K_target,
                experiment_name=experiment_name,
            )
            
            # 5. Synthesis (The Clinical Report)
            return PerformanceAuditor.audit_synthesis(
                q_res,
                fr_res,
                reproducibility=r_info,
                experiment_name=experiment_name,
            )
            
        except baselines.UndefinedBaselineError as e:
            desc = "Clinical scoring could not be completed because no compatible baseline was available."
            return PerformanceAuditor.audit_synthesis(None, None, status=DiagnosticStatus.MISSING_BASELINE, 
                                                   description=desc, reproducibility=r_info, experiment_name=experiment_name)
        except Exception as e:
            raise e
