# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_mop import BaseMop

class DTLZ9(BaseMop):
    """
    DTLZ9 benchmark problem. 
    Constrained problem. N must be a multiple of M.
    """
    def __init__(self, **kwargs):
        m_val = kwargs.get('M', 3)
        if m_val < 2:
            raise ValueError("DTLZ9 requires at least M=2 objectives.")
        
        n_val = kwargs.get('N', None)
        if n_val is None:
            n_val = 10 * m_val
            kwargs['N'] = n_val
            
        if n_val < m_val:
            raise ValueError(
                f"DTLZ9 requires N >= M variables to maintain its block-based mathematical structure.\n"
                f"The problem divides decision variables into M blocks, assigning one block per objective. "
                f"Provided N={n_val}, M={m_val}."
            )
        
        if n_val % m_val != 0:
            import warnings
            warnings.warn(f"DTLZ9 is best defined when N is a multiple of M. "
                          f"Provided N={n_val}, M={m_val}. {n_val % m_val} variables will be ignored.")
        super().__init__(**kwargs)

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

    def ps(self, n_points: int = 100):
        """Analytical sampling of DTLZ9 Pareto Set."""
        M = self.M
        N = self.N
        N_over_M = N // M
        
        # The Pareto Front is on the boundary FJ^2 + FM^2 = 1.
        # To sample the manifold effectively for M objectives:
        res = np.zeros((n_points, N))
        
        for i in range(n_points):
            # 1. Generate a unit vector in the positive quadrant of the M-hypersphere
            # We sample from a normal distribution and take absolute values.
            v = np.abs(np.random.normal(0, 1, M))
            f = v / np.sqrt(np.sum(v**2))
            
            # 2. Map objectives back to decision variables (x = f^10)
            X_row = np.zeros(N)
            for j in range(M):
                start = j * N_over_M
                end = (j + 1) * N_over_M
                X_row[start:end] = f[j]**10
            res[i, :] = X_row
            
        return res