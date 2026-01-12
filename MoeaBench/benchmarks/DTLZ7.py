# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_benchmark import BaseBenchmark

class DTLZ7(BaseBenchmark):
    """
    DTLZ7 benchmark problem. 
    Disconnected Pareto front.
    """
    def __init__(self, M=3, K=20, **kwargs):
        self.K = K
        N = M + K - 1
        super().__init__(M=M, N=N, **kwargs)

    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        M = self.M
        K = self.K
        
        # g = 1 + 9/K * sum(Xi) for i=M to N
        X_m = X[:, M-1:]
        g = 1 + 9 / K * np.sum(X_m, axis=1).reshape(-1, 1)
        
        F = np.zeros((X.shape[0], M))
        # f1..fm-1 = x1..xm-1
        F[:, :M-1] = X[:, :M-1]
        
        # h = M - sum ( fi/(1+g) * (1+sin(3*pi*fi)) )
        h = M - np.sum( (F[:, :M-1] / (1 + g)) * (1 + np.sin(3 * np.pi * F[:, :M-1])), axis=1).reshape(-1, 1)
        
        F[:, M-1:] = (1 + g) * h
        return {'F': F}

    def get_K(self):
        return self.K