# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from itertools import combinations
from .base_mop import BaseMop

class DTLZ8(BaseMop):
    """
    DTLZ8 benchmark problem. 
    Constrained problem with objective-based constraints. 
    N must be a multiple of M.
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
        
        # Objectives: average of variables in blocks
        F = np.zeros((X.shape[0], M))
        for i in range(M):
            start = i * N_over_M
            end = (i + 1) * N_over_M
            F[:, i] = np.mean(X[:, start:end], axis=1)
            
        result = {'F': F}
        
        if n_ieq_constr != 0:
            G = self._calc_constraints(F)
            result['G'] = -G # MoeaBench uses G <= 0 for feasible
            result['feasible'] = np.all(result['G'] <= 0, axis=1)
            
        return result

    def _calc_constraints(self, F):
        M = self.M
        N_samples = F.shape[0]
        
        # gj = fM + 4fj - 1 >= 0
        Gj = F[:, M-1:M] + 4 * F[:, :M-1] - 1
        
        # gM = 2fM + min(fi+fj) - 1 >= 0
        comb = list(combinations(range(M - 1), 2))
        min_sum = np.min(np.column_stack([F[:, c[0]] + F[:, c[1]] for c in comb]), axis=1).reshape(-1, 1)
        Gm = 2 * F[:, M-1:M] + min_sum - 1
        
        return np.concatenate((Gj, Gm), axis=1)

    def get_n_ieq_constr(self):
        return self.M