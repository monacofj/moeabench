# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_dpf import BaseDPF

class DPF1(BaseDPF):
    def ps(self, n_points: int = 100):
        """Preserve native D=2 grid; anchor all essential simplex vertices for D>2."""
        points = super().ps(n_points=n_points)
        if self.D > 2:
            from ._simplex_anchors import anchor_simplex_positions
            anchor_simplex_positions(points[:, :self.D-1], self.D)
        return points

    """
    DPF1 benchmark problem. 
    Flat high-dimensional problem projected from D-dimensional DTLZ1-like base.
    """
    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        D, K, M = self.D, self.K, self.M
        
        # g factor (DTLZ1-like)
        X_m = X[:, D-1:]
        k = X_m.shape[1]
        g = 100 * (k + np.sum((X_m - 0.5)**2 - np.cos(20 * np.pi * (X_m - 0.5)), axis=1)).reshape(-1, 1)
        
        # Base F (D objectives)
        F_base = np.zeros((X.shape[0], D))
        X_front = X[:, :D-1]
        
        for i in range(D):
            f = 0.5 * (1 + g).flatten()
            if i < D - 1:
                f *= np.prod(X_front[:, :D-i-1], axis=1)
            if i > 0:
                f *= (1 - X_front[:, D-i-1])
            F_base[:, i] = f

        return {'F': self._project(F_base)}