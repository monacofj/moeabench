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
    def __init__(self, **kwargs):
        m_val = kwargs.get('M', 3)
        if m_val < 2:
            raise ValueError("DTLZ8 requires at least M=2 objectives.")
        
        n_val = kwargs.get('N', None)
        if n_val is None:
            n_val = 10 * m_val
            kwargs['N'] = n_val
        
        if n_val < m_val:
            raise ValueError(
                f"DTLZ8 requires N >= M variables to maintain its block-based mathematical structure.\n"
                f"The problem divides decision variables into M blocks, assigning one block per objective. "
                f"Provided N={n_val}, M={m_val}."
            )
        
        if n_val % m_val != 0:
            import warnings
            warnings.warn(f"DTLZ8 is best defined when N is a multiple of M. "
                          f"Provided N={n_val}, M={m_val}. {n_val % m_val} variables will be ignored.")
        super().__init__(**kwargs)

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
        # gj = fM + 4fj - 1 >= 0
        Gj = F[:, M-1:M] + 4 * F[:, :M-1] - 1
        
        if M >= 3:
            # gM = 2fM + min(fi+fj) - 1 >= 0
            comb = list(combinations(range(M - 1), 2))
            min_sum = np.min(np.column_stack([F[:, c[0]] + F[:, c[1]] for c in comb]), axis=1).reshape(-1, 1)
            Gm = 2 * F[:, M-1:M] + min_sum - 1
            return np.concatenate((Gj, Gm), axis=1)
        
        return Gj

    def get_n_ieq_constr(self):
        if self.M < 3:
            return 1
        return self.M

    def ps(self, n_points: int = 100):
        """
        Analytical sampling of DTLZ8 Pareto Set.
        Note: DTLZ8 has a complex, constrained topology featuring mixed linear 
        and curved segments. Analytical sampling is not supported in this version.
        """
        raise NotImplementedError("Analytical ps() sampling is not supported for DTLZ8 "
                                  "due to its complex constrained topology.")