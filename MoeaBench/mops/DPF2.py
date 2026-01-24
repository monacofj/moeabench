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
        
        return {'F': self._project(F_base, square=False)}
