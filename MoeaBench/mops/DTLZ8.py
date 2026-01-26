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
        Supports static high-fidelity files for M=3, 5, 10 (Phase X1)
        and Guided Analytical Solver for dynamic M (Phase X2).
        """
        M = self.M
        N = self.N
        
        # Phase X1: Static High-Fidelity lookup
        if M in [3, 5, 10]:
            try:
                import os
                import pandas as pd
                base_path = os.path.dirname(__file__)
                x_file = os.path.join(base_path, "data", f"DTLZ8_{M}_optimal_x.csv")
                if os.path.exists(x_file):
                    df_x = pd.read_csv(x_file, header=None)
                    X = df_x.values
                    if n_points < X.shape[0]:
                        idx = np.linspace(0, X.shape[0] - 1, n_points).astype(int)
                        return X[idx]
                    return X
            except (ImportError, FileNotFoundError):
                pass # Fallback to analytical solver if files missing or pandas unavailable

        # Phase X2: Guided Analytical Solver
        # DTLZ8 Pareto Set is formed by:
        # 1. Symmetric central curve: f_M >= 1/3, f_i = (1 - f_M) / 4
        # 2. Branching arms: f_M < 1/3, where f_i + f_j >= 1 - 2*f_M is active.
        # We sample f_M in [0, 1] and derive other objectives.
        
        points_per_arm = n_points // M
        if points_per_arm < 1: points_per_arm = 1
        
        F_list = []
        
        # 1. Sample Central Curve (f_M from 1/3 to 1)
        fm_central = np.linspace(1/3, 1.0, points_per_arm)
        fi_central = (1.0 - fm_central) / 4.0
        F_central = np.tile(fi_central[:, None], (1, M-1))
        F_central = np.column_stack((F_central, fm_central))
        F_list.append(F_central)
        
        # 2. Sample Branching Arms (f_M from 0 to 1/3)
        # Each arm k (from 0 to M-2) has f_k varying while others compensate.
        fm_arms = np.linspace(0.0, 1/3, points_per_arm, endpoint=False)
        for k in range(M - 1):
            alpha = (1.0 - fm_arms) / 4.0
            beta = (1.0 - 2.0 * fm_arms)
            
            # Sub-sample f_k between alpha and beta/2
            fk = np.linspace(alpha, beta / 2.0, points_per_arm)
            fj = beta - fk
            
            F_arm = np.zeros((points_per_arm, M))
            F_arm[:, M-1] = fm_arms
            # In arm k, f_k varies, and f_j (j != k, M-1) are identical
            for j in range(M - 1):
                F_arm[:, j] = fk if j == k else fj
            F_list.append(F_arm)

        F = np.concatenate(F_list, axis=0)
        
        # 3. Map F -> X
        # In DTLZ8, fi = mean(X_block_i). Minimal norm solution is constant block.
        N_over_M = N // M
        X = np.zeros((F.shape[0], N))
        for i in range(M):
            start = i * N_over_M
            end = (i + 1) * N_over_M
            X[:, start:end] = F[:, i:i+1]
            
        return X