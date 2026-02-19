# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, Union

class BaseMop(ABC):
    """
    Abstract Base Class for all Multi-Objective Problems (MOPs) in MoeaBench.
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
        Should be overridden by subclasses.
        """
        raise NotImplementedError(f"ps() sampling (analytical) not implemented for {self.__class__.__name__}")

    def calibrate(self, baseline: Optional[str] = None, force: bool = False, **kwargs) -> bool:
        """
        Calibrates the clinical diagnostics for this MOP instance.
        Generates/Loads a sidecar JSON file with Ground Truth and Baselines.
        """
        from ..diagnostics.calibration import calibrate_mop
        return calibrate_mop(self, baseline=baseline, force=force, **kwargs)
