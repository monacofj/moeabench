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
        Analyzes a dictionary of metrics (IGD, GD, H_rel, EMD) and returns a diagnosis.
        """
        # Extract core metrics with safe defaults
        igd = metrics.get('igd', metrics.get('IGD', float('inf')))
        gd = metrics.get('gd', metrics.get('GD', float('inf')))
        h_rel = metrics.get('h_rel', metrics.get('H_rel', 0.0))
        emd = metrics.get('emd', metrics.get('EMD', 0.0))
        
        # 1. Check for Super-Saturation (Resolution Artifacts)
        if h_rel > 1.0:
            return DiagnosticResult(
                status=DiagnosticStatus.SUPER_SATURATION,
                metrics=metrics,
                confidence=1.0,
                _rationale=(
                    f"Super-Saturation Detected (H_rel = {h_rel*100:.2f}%). "
                    "The algorithm has identified a discrete distribution that exceeds the "
                    "volume of the sampled Ground Truth. This indicates extreme performance "
                    "beyond the resolution of the reference baseline."
                )
            )

        # 2. Check for Diversity Collapse (The "DPF2 Paradox")
        # Low GD (close to front) but High IGD (poor coverage)
        if gd < 0.01 and igd > 0.1:
            return DiagnosticResult(
                status=DiagnosticStatus.DIVERSITY_COLLAPSE,
                metrics=metrics,
                confidence=0.95,
                _rationale=(
                    f"Diversity Collapse Detected. The algorithm exhibits excellent convergence "
                    f"(GD={gd:.1e}) but poor coverage (IGD={igd:.1e}). "
                    "This indicates the population has degenerated into a small subset "
                    "or a single point on the Pareto Front."
                )
            )

        # 3. Check for Topological Distortion
        if emd > 0.08:
            return DiagnosticResult(
                status=DiagnosticStatus.TOPOLOGICAL_DISTORTION,
                metrics=metrics,
                confidence=0.8,
                _rationale=(
                    f"Topological Distortion Detected (EMD={emd:.3f}). "
                    "While convergence metrics may be acceptable, the manifold reconstruction "
                    "error is high, suggesting the algorithm failed to capture the true "
                    "geometry/curvature of the Pareto Front."
                )
            )
            
        # 4. Check for Convergence Failure
        if gd > 0.1 and igd > 0.1:
            return DiagnosticResult(
                status=DiagnosticStatus.CONVERGENCE_FAILURE,
                metrics=metrics,
                confidence=0.9,
                _rationale=(
                    f"Convergence Failure. Both convergence (GD={gd:.1e}) and coverage "
                    f"(IGD={igd:.1e}) are insufficient. The algorithm likely failed to "
                    "approach the Pareto Front effectively."
                )
            )

        # 5. Default to Optimal
        if igd <= 0.1 and gd <= 0.1:
             return DiagnosticResult(
                status=DiagnosticStatus.OPTIMAL,
                metrics=metrics,
                confidence=0.9,
                _rationale=(
                    f"Optimal Performance. The algorithm achieved a balanced trade-off "
                    f"between convergence (GD={gd:.1e}) and diversity (IGD={igd:.1e}), "
                    f"recovering {h_rel*100:.1f}% of the reference volume."
                )
            )

        return DiagnosticResult(
            status=DiagnosticStatus.UNDEFINED,
            metrics=metrics,
            confidence=0.0,
            _rationale="Insufficient data or ambiguous metric signature for automated diagnosis."
        )

def audit(target: Any, ground_truth: Optional[Any] = None) -> DiagnosticResult:
    """
    Smart delegate for performance auditing.
    
    Args:
        target: Can be a dictionary of metrics, an Experiment, or a Result object.
        ground_truth: Optional reference front (if target contains raw populations).
    """
    metrics = {}
    
    # Smart Delegation Logic
    if isinstance(target, dict):
        metrics = target
    # Add more type handlers here (e.g. for Experiment or Result objects calculation logic)
    
    return PerformanceAuditor.audit(metrics)
