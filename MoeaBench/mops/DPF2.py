# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_dpf import BaseDPF

class DPF2(BaseDPF):
    """
    DPF2 benchmark problem.
    Flat high-dimensional problem projected from D-dimensional DTLZ7-like base.
    """
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

    def ps(self, n_points=100):
        """Analytical sampling of DPF2 Pareto Set (DTLZ7-like g=1)."""
        D = self.D
        N = self.N
        res = np.zeros((n_points, N))
        res[:, :D-1] = np.random.random((n_points, D - 1))
        # Optimal g=1 when xi=0 for i >= D-1
        res[:, D-1:] = 0.0
        return res
