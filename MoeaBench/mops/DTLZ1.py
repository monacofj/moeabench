# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_mop import BaseMop

class DTLZ1(BaseMop):
    """
    DTLZ1 benchmark problem.
    """
    def __init__(self, M=3, K=5, **kwargs):
        self.K = K
        N = M + K - 1
        super().__init__(M=M, N=N, **kwargs)

    def evaluation(self, X, n_ieq_constr=0):
        """
        Standard DTLZ1 evaluation.
        g = 100 * [K + sum( (xi - 0.5)^2 - cos(20*pi*(xi-0.5)) )]
        f1 = 0.5 * x1 * x2 * ... * (1 + g)
        ...
        fm = 0.5 * (1 - x1) * (1 + g)
        """
        X = np.atleast_2d(X)
        M = self.M
        
        # g = 100 * (K + sum((xi - 0.5)**2 - cos(20 * pi * (xi - 0.5))))
        X_m = X[:, M-1:]
        g = 100 * (self.K + np.sum((X_m - 0.5)**2 - np.cos(20 * np.pi * (X_m - 0.5)), axis=1)).reshape(-1, 1)
        
        F = np.zeros((X.shape[0], M))
        
        X_front = X[:, :M-1]
        
        for i in range(M):
            f = 0.5 * (1 + g).flatten()
            if i < M - 1:
                f *= np.prod(X_front[:, :M-i-1], axis=1)
            
            if i > 0:
                f *= (1 - X_front[:, M-i-1])
                
            F[:, i] = f
            
        return {'F': F}

    def ps(self, n_points=100):
        """Analytical sampling of DTLZ1 Pareto Set."""
        M = self.M
        N = self.N
        res = np.zeros((n_points, N))
        res[:, :M-1] = np.random.random((n_points, M - 1))
        res[:, M-1:] = 0.5
        return res

    def get_K(self):
        return self.K