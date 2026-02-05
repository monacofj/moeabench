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

    @property
    def gd(self) -> float:
        """Convenience access to Generational Distance."""
        return self.metrics.get('gd', float('inf'))

    @property
    def igd(self) -> float:
        """Convenience access to Inverted Generational Distance."""
        return self.metrics.get('igd', float('inf'))

    @property
    def emd(self) -> float:
        """Convenience access to Earth Mover's Distance."""
        return self.metrics.get('emd', float('inf'))

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
        emd = metrics.get('emd', metrics.get('EMD', float('inf')))
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
                    "This indicates performance saturation strictly superior to the discrete baseline."
                )
            )

        # 3. Binary Classification (The Truth Table)
        # Thresholds defined in ADR 0025
        # GOOD_GD (< 0.1)  : Converged?
        # GOOD_IGD (< 0.1) : Covered?
        # GOOD_EMD (< 0.12): Shape/Distribution Fidelity? (Relaxed from 0.08)
        GOOD_GD = gd < 0.1
        GOOD_IGD = igd < 0.1
        GOOD_EMD = emd < 0.12

        # State Determination (8 States)
        if GOOD_GD:
            if GOOD_IGD:
                if GOOD_EMD:
                    # (1) [L, L, L] -> IDEAL FRONT
                    return DiagnosticResult(
                        status=DiagnosticStatus.IDEAL_FRONT, metrics=metrics, confidence=0.98,
                        _rationale=f"Ideal Front. The population lies close to the ground-truth Pareto front (GD={gd:.1e}), covers the full extent of the reference front without relevant gaps (IGD={igd:.2f}), and matches the reference distribution in a global, mass-transport sense (EMD={emd:.3f}). In practical terms, you not only 'hit the target,' but you also covered it uniformly and with the expected density."
                    )
                else:
                    # (2) [L, L, H] -> BIASED SPREAD
                    return DiagnosticResult(
                        status=DiagnosticStatus.BIASED_SPREAD, metrics=metrics, confidence=0.85,
                        _rationale=f"Biased Spread. Solutions are near the true front (GD={gd:.1e}) and the front is broadly covered (IGD={igd:.2f}), yet the global distribution is distorted (EMD={emd:.3f}). This typically means the algorithm concentrates too much mass in some regions and too little in others—clustering or systematic density bias."
                    )
            else:
                if GOOD_EMD:
                    # (3) [L, H, L] -> GAPPED COVERAGE
                    return DiagnosticResult(
                        status=DiagnosticStatus.GAPPED_COVERAGE, metrics=metrics, confidence=0.88,
                        _rationale=f"Gapped Coverage. The solutions that exist are close to the true front (GD={gd:.1e}), but substantial regions of the reference front have no nearby representatives (IGD={igd:.2f}). Despite local correctness (EMD={emd:.3f}), the approximation captures only parts of the Pareto set, leaving 'holes'."
                    )
                else:
                    # (4) [L, H, H] -> COLLAPSED FRONT
                    return DiagnosticResult(
                        status=DiagnosticStatus.COLLAPSED_FRONT, metrics=metrics, confidence=0.95,
                        _rationale=f"Collapsed Front. The population is largely near the front (GD={gd:.1e}) but fails to represent its extent (IGD={igd:.2f}) and does not match the reference distribution globally (EMD={emd:.3f}). This is the signature of degeneracy: the algorithm collapses onto a small subset of the front (or even a single point)."
                    )
        else: # BAD GD
            if GOOD_IGD:
                if GOOD_EMD:
                    # (5) [H, L, L] -> NOISY POPULATION
                    return DiagnosticResult(
                        status=DiagnosticStatus.NOISY_POPULATION, metrics=metrics, confidence=0.80,
                        _rationale=f"Noisy Population. The reference front is well covered (IGD={igd:.2f}) and the global distribution resembles the reference (EMD={emd:.3f}), yet the population contains substantial mass far from the true front (GD={gd:.1e}). Scientifically, this indicates low purity: a meaningful subset approximates the front well, but many dominated points remain."
                    )
                else:
                    # (6) [H, L, H] -> DISTORTED COVERAGE
                    return DiagnosticResult(
                        status=DiagnosticStatus.DISTORTED_COVERAGE, metrics=metrics, confidence=0.82,
                        _rationale=f"Distorted Coverage. Coverage exists in the nearest-neighbor sense (IGD={igd:.2f}), but the population is globally inconsistent: it is far on average from the true front (GD={gd:.1e}) and its distribution deviates strongly (EMD={emd:.3f}). This often reflects a mixed regime with severe bias or contamination."
                    )
            else:
                if GOOD_EMD:
                    # (7) [H, H, L] -> SHIFTED_FRONT
                    return DiagnosticResult(
                        status=DiagnosticStatus.SHIFTED_FRONT, metrics=metrics, confidence=0.90,
                        _rationale=f"Shifted Front. The population forms a coherent, well-structured front-like manifold (EMD={emd:.3f}), but it is displaced relative to the true Pareto front (GD={gd:.1e}). Intuitively, the algorithm learned the 'right shape' but at the 'wrong location': a systematic mislocalization (e.g. Local Optimum Trap). For completeness, IGD={igd:.2f}."
                    )
                else:
                    # (8) [H, H, H] -> SEARCH_FAILURE
                    return DiagnosticResult(
                        status=DiagnosticStatus.SEARCH_FAILURE, metrics=metrics, confidence=0.95,
                        _rationale=f"Search Failure. The population is neither close to the true front (GD={gd:.1e}) nor does it cover it (IGD={igd:.2f}), and its global distribution is unrelated to the reference. A full failure mode. For completeness, EMD={emd:.3f}."
                    )

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
