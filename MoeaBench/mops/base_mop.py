# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, Union

class BaseMop(ABC):
    """
    Abstract Base Class for all Multi-Objective Problems (MOPs) in MoeaBench.
    """
    def __init__(self, M: int = 3, N: Optional[int] = None, 
                 xl: Optional[Any] = None, xu: Optional[Any] = None, **kwargs: Any) -> None:
        self.M = M
        self.N = N
        
        # Lower and upper bounds
        if xl is not None:
            self.xl = np.array(xl)
        else:
            self.xl = np.zeros(self.N) if self.N else None
            
        if xu is not None:
            self.xu = np.array(xu)
        else:
            self.xu = np.ones(self.N) if self.N else None

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
