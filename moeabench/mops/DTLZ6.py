# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .DTLZ5 import DTLZ5

class DTLZ6(DTLZ5):
    """
    DTLZ6 benchmark problem.
    Biased degenerate spherical problem.
    """
    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        M = self.M
        X_m = X[:, M-1:]
        
        # g = sum(xi^0.1)
        g = np.sum(X_m**0.1, axis=1).reshape(-1, 1)
        
        # Theta logic same as DTLZ5
        theta = [X[:, 0:1] * np.pi/2]
        for i in range(1, M-1):
            theta.append((np.pi / (4 * (1 + g))) * (1 + 2 * g * X[:, i:i+1]))
            
        return self._spherical_evaluation(X, g, theta=theta)
            
    def ps(self, n_points=100):
        """Analytical sampling of DTLZ6 Pareto Set."""
        M = self.M
        N = self.N
        res = np.zeros((n_points, N))
        res[:, :M-1] = np.random.random((n_points, M - 1))
        # Optimal g=0 when xi=0 for i >= M
        res[:, M-1:] = 0.0
        return res