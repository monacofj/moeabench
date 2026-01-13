# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_mop import BaseMop

class DTLZ2(BaseMop):
    """
    DTLZ2 benchmark problem.
    """
    def __init__(self, M=3, K=10, **kwargs):
        self.K = K
        N = M + K - 1
        super().__init__(M=M, N=N, **kwargs)

    def evaluation(self, X, n_ieq_constr=0):
        """
        Standard DTLZ2 evaluation.
        """
        X = np.atleast_2d(X)
        M = self.M
        
        # g = sum ( (xi - 0.5)^2 ) for i = M to N
        X_m = X[:, M-1:]
        g = np.sum((X_m - 0.5)**2, axis=1).reshape(-1, 1)
        
        return self._spherical_evaluation(X, g)

    def _spherical_evaluation(self, X, g, theta=None):
        M = self.M
        F = np.zeros((X.shape[0], M))
        
        if theta is None:
            # Standard DTLZ2-4 theta
            theta = X[:, :M-1] * (np.pi / 2)
        elif isinstance(theta, list):
            # DTLZ5-6 return list of columns
            theta = np.column_stack(theta)
            
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        for i in range(M):
            f = (1 + g).flatten()
            if i < M - 1:
                f *= np.prod(cos_theta[:, :M-i-1], axis=1)
            
            if i > 0:
                f *= sin_theta[:, M-i-1]
                
            F[:, i] = f
        return {'F': F}

    def get_K(self):
        return self.K