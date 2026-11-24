# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, Union
from ..core.base import Reportable

class BaseMop(Reportable, ABC):
    """
    Abstract Base Class for all Multi-Objective Problems (MOPs) in moeabench.
    """
    def __init__(self, **kwargs: Any) -> None:
        self.name = kwargs.pop('name', self.__class__.__name__)
        self.M = kwargs.pop('M', 3)
        self.N = kwargs.pop('N', None)
        self.kwargs = kwargs
        
        # Lower and upper bounds
        xl = kwargs.pop('xl', None)
        if xl is not None:
            self.xl = np.array(xl)
        else:
            self.xl = np.zeros(self.N) if self.N else None
            
        xu = kwargs.pop('xu', None)
        if xu is not None:
            self.xu = np.array(xu)
        else:
            self.xu = np.ones(self.N) if self.N else None

        # Validation hook
        self.validate()

    def validate(self):
        """
        Validation hook to be overridden by subclasses to ensure 
        mathematical consistency of input parameters.
        """
        if self.N is not None and self.N < 1:
            raise ValueError(f"Number of variables (N) must be at least 1, got {self.N}")
        if self.M < 2:
            raise ValueError(f"Number of objectives (M) must be at least 2, got {self.M}")

    @abstractmethod
    def evaluation(self, X: np.ndarray, n_ieq_constr: int = 0) -> Dict[str, np.ndarray]:
        """
        Evaluates a set of decision variables.
        
        Args:
            X (np.ndarray): Matrix of decision variables (N_samples x N_vars).
            n_ieq_constr (int): Number of inequality constraints to evaluate.
            
        Returns:
            dict: { 'F': np.ndarray, 'G': np.ndarray (optional) }
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> 'BaseMop':
        """Compatibility with legacy factory call pattern."""
        return self

    def get_M(self) -> int:
        return self.M

    def get_Nvar(self) -> Optional[int]:
        return self.N
    
    def get_n_ieq_constr(self) -> int:
        # Default to 0, can be overridden
        return 0

    def pf(self, n_points: int = 100) -> np.ndarray:
        """
        Samples the true Pareto Front (objectives).
        By default, evaluates the analytical Pareto Set samples (ps).
        """
        ps_samples = self.ps(n_points)
        res = self.evaluation(ps_samples)
        return res['F']

    def ps(self, n_points: int = 100) -> np.ndarray:
        """
        Samples the true Pareto Set (decision variables).
        
        The analytical Pareto-set sample can be used to construct reference fronts.
        
        Args:
            n_points (int): Number of target points to sample. 
                            For M=2, usually a 1D linear space.
                            For M=3, usually a 2D grid/triangulation.
        """
        raise NotImplementedError(f"ps() sampling (analytical) not implemented for {self.__class__.__name__}")

    def calibrate(
        self,
        source_baseline: Optional[str] = None,
        source_gt: Optional[Union[str, np.ndarray]] = None,
        source_search: Optional[Any] = None,
        force: bool = False,
        **kwargs,
    ) -> bool:
        """Calibrate this MOP's Ground Truth independently of FAIR.

        Historical arguments for the three GT sources, portable sidecar and
        reload/force behavior are preserved. Experimental ECDF-only parameters
        are unsupported in the GT-only implementation.
        """
        from .calibration import calibrate_mop
        return calibrate_mop(
            self,
            source_baseline=source_baseline,
            source_gt=source_gt,
            source_search=source_search,
            force=force,
            **kwargs,
        )

    @property
    def calibrated_gt(self) -> np.ndarray:
        """Return a copy of this MOP's calibrated objective-space truth."""
        from .calibration import calibrated_gt
        return calibrated_gt(self)

    def assess_gt(self, reference=None, *, n_points=None, probes=128,
                  seed=0, tolerance=1e-9, dominance_margin=None,
                  dominance_max_intervals=4096,
                  dominance_g_gap=None, dominance_g_max_intervals=8192,
                  decision_reference=None, local_descent=False,
                  objective_only_local_descent=False):
        """Read-only structured GT assessment (ADR 0059).

        Use the calibrated in-memory GT by default, an explicit objective
        matrix via reference, or an on-demand analytical sample via n_points.
        Never calibrates or writes a reference implicitly.
        """
        from .gt_assessment import assess_gt
        return assess_gt(
            self, reference=reference, n_points=n_points,
            probes=probes, seed=seed, tolerance=tolerance,
            dominance_margin=dominance_margin,
            dominance_max_intervals=dominance_max_intervals,
            dominance_g_gap=dominance_g_gap,
            dominance_g_max_intervals=dominance_g_max_intervals,
            decision_reference=decision_reference,
            local_descent=local_descent,
            objective_only_local_descent=objective_only_local_descent)

    def report(self, show: bool = True, *, assess: bool = True,
               probes: int = 128, seed: int = 0,
               dominance_margin=None, dominance_max_intervals=4096,
               dominance_g_gap=None, dominance_g_max_intervals=8192,
               decision_reference=None, local_descent=False,
               objective_only_local_descent=False,
               **kwargs) -> str:
        """MOP definition and the current calibrated GT's quality context.

        Existing in-memory calibration is assessed read-only; this report
        never generates a new GT, reads sidecars or runs a hidden
        calibration. Explicitly pass assess=False for a cheap MOP summary.
        """
        lines = [
            f"### Multiobjective Problem: {self.name}",
            "",
            f"Objectives (M): {self.M}",
            f"Decision variables (N): {self.N}",
            f"Constraints: {self.get_n_ieq_constr()} inequalities",
            "",
            "### Ground Truth",
        ]
        gt = getattr(self, "_gt_reference", None)
        if gt is None:
            lines.append("Not calibrated on this MOP instance. "
                         "Call mop.calibrate() explicitly.")
        elif not assess:
            lines.append("Calibrated GT present; assessment not requested.")
        else:
            assessment = self.assess_gt(
                probes=probes, seed=seed,
                dominance_margin=dominance_margin,
                dominance_max_intervals=dominance_max_intervals,
                dominance_g_gap=dominance_g_gap,
                dominance_g_max_intervals=dominance_g_max_intervals,
                decision_reference=decision_reference,
                local_descent=local_descent,
                objective_only_local_descent=objective_only_local_descent)
            lines.append("Quality diagnostics (not a proof of full coverage):")
            lines.append(self._report_block(assessment.summary()))
        lines.append("")
        return self._render_report("\n".join(lines), show=show, **kwargs)
