# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from itertools import accumulate, repeat, cycle, islice
from .base_mop import BaseMop

class BaseDPF(BaseMop):
    """
    Shared logic for Dominance-Preserving Flattening (DPF) benchmarks.
    """
    def __init__(self, M=3, D=2, K=5, **kwargs):
        if D < 2 or D >= M:
            D = 2
        self.D = D
        self.K = K
        N = D + K - 1
        super().__init__(M=M, N=N, **kwargs)
        self._chaos_weights = self._calc_chaos_weights()

    def _calc_chaos_weights(self):
        U = 1
        factor = (self.M - self.D) * self.D
        P = factor
        VET = list(accumulate(repeat(None, factor), lambda acc, _: 3.8 * acc * (1 - acc), initial=0.1))[1:]
        V_CYCLE = cycle(VET)
        CHAOS = np.array(sorted(list(islice(V_CYCLE, P * U))))
        return CHAOS.reshape(P, U)

    def _calc_dynamic_chaos(self, n_samples):
        U = self.M - self.D
        P = n_samples
        if U <= 0: return np.zeros((P, 0))
        
        # Logistic map logic matching NU_chaos exactly
        # Note: NU_chaos uses a factor for VET size, but here we just need P*U samples
        VET = list(accumulate(repeat(None, P*U), lambda acc, _: 3.8 * acc * (1 - acc), initial=0.1))[1:]
        # NU_chaos sorts them!
        CHAOS = np.array(sorted(VET))
        return CHAOS.reshape(P, U)

    def _project(self, F_base, square=False):
        if self.M == self.D:
            return F_base
        redundant = []
        for row in range(0, self._chaos_weights.shape[0], self.D):
            weights = self._chaos_weights[row : row + self.D, :]
            proj = np.dot(F_base, weights)
            if square:
                proj = proj**2
            redundant.append(proj)
        return np.concatenate((F_base, np.column_stack(redundant)), axis=1)

    def get_D(self):
        return self.D

    def get_K(self):
        return self.K
