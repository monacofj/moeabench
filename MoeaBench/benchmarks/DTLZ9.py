# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_benchmark import BaseBenchmark

class DTLZ9(BaseBenchmark):
    """
    DTLZ9 benchmark problem. 
    Constrained problem. N must be a multiple of M.
    """
    def __init__(self, M=3, N=None, **kwargs):
        if N is None:
            N = 10 * M
        super().__init__(M=M, N=N, **kwargs)

    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        M = self.M
        N = self.N
        N_over_M = N // M
        
        # Objectives: sum(xi^0.1) in blocks
        F = np.zeros((X.shape[0], M))
        for i in range(M):
            start = i * N_over_M
            end = (i + 1) * N_over_M
            F[:, i] = np.mean(X[:, start:end]**0.1, axis=1)
            
        result = {'F': F}
        
        if n_ieq_constr != 0:
            G = self._calc_constraints(F)
            result['G'] = -G # MoeaBench uses G <= 0 for feasible
            result['feasible'] = np.all(result['G'] <= 0, axis=1)
            
        return result

    def _calc_constraints(self, F):
        M = self.M
        # gj = (fM^2 + fj^2) - 1 >= 0
        Gj = (F[:, M-1:M]**2) + (F[:, :M-1]**2) - 1
        return Gj

    def get_n_ieq_constr(self):
        return self.M - 1