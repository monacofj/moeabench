import numpy as np
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from .enums import DiagnosticStatus

@dataclass
class DiagnosticResult:
    """
    Encapsulates the finding of an algorithmic audit.
    """
    status: DiagnosticStatus
    metrics: Dict[str, float]
    confidence: float
    _rationale: str

    def rationale(self) -> str:
        """Returns the scientific-didactic explanation of the diagnosis."""
        return self._rationale

    def report(self, **kwargs) -> str:
        """Formatted narrative report."""
        lines = [
            f"--- [Diagnostics] Algorithmic Pathology Report ---",
            f"  Status: {self.status.name}",
            f"  Confidence: {self.confidence:.2f}",
            f"  Analysis: {self._rationale}",
            "--------------------------------------------------"
        ]
        return "\n".join(lines)

    def report_show(self, **kwargs) -> None:
        """Displays the report in the appropriate environment (Terminal/Notebook)."""
        try:
            from IPython.display import Markdown, display
            # Check if running in a notebook environment
            import sys
            if 'ipykernel' in sys.modules:
                display(Markdown(f"""
### 🩺 Algorithmic Pathology Report
**Status**: `{self.status.name}` (Confidence: {self.confidence:.2f})

> **Analysis**: {self._rationale}
"""))
                return
        except ImportError:
            pass
        
        # Fallback to terminal print
        print(self.report())

class PerformanceAuditor:
    """
    Expert system for interpreting multi-objective performance metrics.
    """
    
    @staticmethod
    def audit(metrics: Dict[str, float]) -> DiagnosticResult:
        """
        Analyzes a dictionary of metrics (IGD, GD, H_rel, EMD) using the 8-state Pathology Truth Table.
        """
        # 1. Extract Metrics
        igd = metrics.get('igd', metrics.get('IGD', float('inf')))
        gd = metrics.get('gd', metrics.get('GD', float('inf')))
        emd = metrics.get('emd', metrics.get('EMD', float('inf'))) # Default to inf if not computed
        h_rel = metrics.get('h_rel', metrics.get('H_rel', 0.0))

        # 2. Check for Super-Saturation (Pre-check)
        if h_rel > 1.05: # 5% tolerance
            return DiagnosticResult(
                status=DiagnosticStatus.SUPER_SATURATION,
                metrics=metrics,
                confidence=1.0,
                _rationale=(
                    f"Super-Saturation Detected (H_rel = {h_rel*100:.2f}%). "
                    "The algorithm has outperformed the resolution of the provided Ground Truth. "
                )
            )

        # 3. Binary Classification (The Truth Table)
        # Thresholds defined in ADR 0025
        GOOD_GD = gd < 0.1  # Convergence threshold (standardized for practical runs)
        GOOD_IGD = igd < 0.1 # Coverage threshold
        GOOD_EMD = emd < 0.08 # Topology threshold

        # State Determination
        if GOOD_GD:
            if GOOD_IGD:
                if GOOD_EMD:
                    # (B, B, B) -> OPTIMAL
                    return DiagnosticResult(
                        status=DiagnosticStatus.OPTIMAL, metrics=metrics, confidence=0.95,
                        _rationale=f"Optimal Performance. The algorithm found a well-distributed prediction of the Pareto Front (GD={gd:.1e}, IGD={igd:.2f})."
                    )
                else:
                    # (B, B, R) -> DISTRIBUTION BIAS
                    return DiagnosticResult(
                        status=DiagnosticStatus.DISTRIBUTION_BIAS, metrics=metrics, confidence=0.85,
                        _rationale=f"Distribution Bias. The front is well-converged and covered, but the internal distribution is irregular (EMD={emd:.3f}). This suggests diversity operators are fighting the manifold geometry."
                    )
            else:
                if GOOD_EMD:
                    # (B, R, B) -> SPARSE APPROXIMATION
                    return DiagnosticResult(
                        status=DiagnosticStatus.SPARSE_APPROXIMATION, metrics=metrics, confidence=0.80,
                        _rationale=f"Sparse Approximation. The solutions found are optimal (GD={gd:.1e}) and topologically correct (EMD={emd:.3f}), but too few to ensure global coverage (IGD={igd:.2f})."
                    )
                else:
                    # (B, R, R) -> DIVERSITY COLLAPSE
                    return DiagnosticResult(
                        status=DiagnosticStatus.DIVERSITY_COLLAPSE, metrics=metrics, confidence=0.95,
                        _rationale=f"Diversity Collapse. Excellent convergence (GD={gd:.1e}) but poor coverage and topology. The population has likely degenerated into a single point or small cluster."
                    )
        else: # BAD GD
            if GOOD_IGD:
                # (R, B, X) -> METRIC CONTRADICTION
                # Impossible to have Good Coverage (IGD) if Convergence (GD) is bad, 
                # because IGD is lower-bounded by GD distance.
                return DiagnosticResult(
                    status=DiagnosticStatus.METRIC_CONTRADICTION, metrics=metrics, confidence=1.0,
                    _rationale=f"Metric Contradiction. It is mathematically impossible to have good coverage (IGD={igd:.2f}) with poor convergence (GD={gd:.1e}). Check your Ground Truth data integrity."
                )
            else:
                if GOOD_EMD:
                    # (R, R, B) -> SHADOW FRONT
                    return DiagnosticResult(
                        status=DiagnosticStatus.SHADOW_FRONT, metrics=metrics, confidence=0.75,
                        _rationale=f"Shadow Front Detected. The algorithm failed to converge (GD={gd:.1e}), but the shape of the local front matches the true front (EMD={emd:.3f}). It may be trapped in a local Pareto Front translation."
                    )
                else:
                    # (R, R, R) -> CONVERGENCE FAILURE
                    return DiagnosticResult(
                        status=DiagnosticStatus.CONVERGENCE_FAILURE, metrics=metrics, confidence=0.90,
                        _rationale=f"Convergence Failure. The algorithm failed to converge, cover, or capture the shape of the problem effectively (GD={gd:.1e}, IGD={igd:.2f})."
                    )

def audit(target: Any, ground_truth: Optional[Any] = None) -> DiagnosticResult:
    """
    Smart delegate for performance auditing.
    
    Args:
        target: Can be a dictionary of metrics, an Experiment, or a Result object.
        ground_truth: Optional reference front (if target contains raw populations).
    """
def audit(target: Any, ground_truth: Optional[Any] = None) -> DiagnosticResult:
    """
    Smart delegate for performance auditing.
    
    Args:
        target: Can be a dictionary of metrics, an Experiment, or a Run object.
        ground_truth: Optional reference front (if target contains raw populations).
    """
    metrics_data = {}
    
    # 1. Handle Dictionaries
    if isinstance(target, dict):
        metrics_data = target
    
    # 2. Handle Experiments and Runs
    else:
        # Import core objects to avoid circularity
        from ..core.experiment import experiment
        from ..core.run import Run
        from .. import metrics as mb_metrics
        
        pop_objs = None
        prob = None

        if isinstance(target, experiment):
            if not target.runs:
                return PerformanceAuditor.audit({}) # Undefined
            pop_objs = target.last_pop.objectives
            prob = target.mop
        elif isinstance(target, Run):
            pop_objs = target.last_pop.objectives
            prob = target.experiment.mop if target.experiment else None
            
        if pop_objs is not None:
             # Identify Ground Truth
             pf = ground_truth
             if pf is None and prob is not None:
                 if hasattr(prob, 'pf'):
                     pf = prob.pf()
                 elif hasattr(prob, 'optimal_front'):
                     pf = prob.optimal_front()
             
             if pf is not None:
                  # Use public API for robust calculation
                  metrics_data['gd'] = float(mb_metrics.gd(pop_objs, ref=pf))
                  metrics_data['igd'] = float(mb_metrics.igd(pop_objs, ref=pf))
                  try:
                      metrics_data['h_rel'] = float(mb_metrics.hv(pop_objs, ref=pf))
                  except:
                      pass
                      
    return PerformanceAuditor.audit(metrics_data)
