# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_dpf import BaseDPF
from ._dtlz7_pareto_regions import sample_dtlz7_record_parameters

class DPF2(BaseDPF):
    """
    DPF2 benchmark problem.
    Flat high-dimensional problem projected from D-dimensional DTLZ7-like base.
    """
    _optimal_g_val = 0.0

    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        D, K, M = self.D, self.K, self.M
        
        # g factor (DTLZ7-like)
        X_m = X[:, D-1:]
        k = X_m.shape[1]
        g = 1 + 9/k * np.sum(X_m, axis=1).reshape(-1, 1)
        
        # Base F (D objectives)
        F_base = np.zeros((X.shape[0], D))
        F_base[:, :D-1] = X[:, :D-1]
        
        # h factor
        h = D - np.sum((F_base[:, :D-1] / (1 + g)) * (1 + np.sin(3 * np.pi * F_base[:, :D-1])), axis=1).reshape(-1, 1)
        F_base[:, D-1:] = (1 + g) * h
        
        return {'F': self._project(F_base, square=True)}

    def ps(self, n_points: int = 100):
        """Sample the genuine DPF2 Pareto set without dominated base points.

        For D=2, retain a deterministic decision-parameter grid, but map
        it onto the two exact DTLZ7-like record intervals. For D>2,
        sample each allowed parameter with the historical private seed.
        The nonnegative monotone redundant projection leaves the base
        Pareto ordering unchanged. The law is parameter-uniform, not
        objective-surface-area uniform.
        """
        points = super().ps(n_points=n_points)
        points[:, :self.D-1] = sample_dtlz7_record_parameters(
            len(points), self.D-1,
            seed=42, structured=(self.D == 2),
        )
        return points
