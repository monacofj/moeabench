# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_dpf import BaseDPF

class DPF5(BaseDPF):
    """
    DPF5 benchmark problem.
    Conditional evaluation based on x1 > 1/3.
    Use distinct function sets "P" and "D".
    """
    def _calc_theta(self, X):
        return X * (np.pi / 2) # Linear mapping assumed from legacy calc_TH(X)

    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        D, K, M = self.D, self.K, self.M
        
        # g = (x_M - x_1)^2 + sum((x_i - 0.5)^2) for i=M to N
        # Note: indices in legacy are 1-based logic?
        # Legacy K_DPF5 calc_g:
        # ((X[:,M-1:M]-X[:,0:1])**2) + sum(...)
        # X[:, M-1] is x_M? No, x_{M-1}? (0-based M-1 is Mth variable)
        # Yes, Mth variable minus First variable.
        g = ((X[:, M-1:M] - X[:, 0:1])**2) + np.sum((X[:, M:] - 0.5)**2, axis=1).reshape(-1, 1)
        
        theta = self._calc_theta(X) # All theta
        # Helper aliases for theta slices used in definitions
        # NOTE: Legacy defines specific ranges [0:M-2] etc.
        
        # Helper to compute prod(cos)
        def prod_cos(start, end):
            if end <= start: return 1.0
            return np.prod(np.cos(theta[:, start:end]), axis=1).reshape(-1, 1)

        def sin_th(idx):
            return np.sin(theta[:, idx:idx+1])
            
        def cos_th(idx):
            return np.cos(theta[:, idx:idx+1])

        # Define 7 kernels (B1..Yd)
        # B1: prod(cos(0:M-2)) * cos(M-2)
        k_B1 = prod_cos(0, M-2) * cos_th(M-2)
        
        # B2: prod(cos(0:M-2)) * sin(M-2)
        k_B2 = prod_cos(0, M-2) * sin_th(M-2)
        
        # Bmd1: prod(cos(0:D-1)) * sin(D-1)
        k_Bmd1 = prod_cos(0, D-1) * sin_th(D-1)
        
        # Y1: sqrt(1/(M-D+1)) * prod(cos(0:D-1))
        # Wait, sqrt factor? Legacy Y1 line 33: np.sqrt(1/(M-D+1))*...
        k_Y1 = np.sqrt(1/(M-D+1)) * prod_cos(0, D-1)
        
        # Y2: prod(cos(0:D-2)) * sin(D-1)
        # Wait, legacy Y2 line 37: prod(cos(X[0:D-2]))*np.sin(X[D-1:D])
        k_Y2 = prod_cos(0, D-2) * sin_th(D-1)
        
        # Yd1: cos(0) * sin(1)
        k_Yd1 = cos_th(0) * sin_th(1)
        
        # Yd: sin(0)
        k_Yd = sin_th(0)
        
        # Prepare Objective Matrices
        F_P = np.zeros((X.shape[0], M))
        F_D = np.zeros((X.shape[0], M))

        # Replicate Legacy logic (calc_F_P) for Set P
        # Fi goes from 1 to M
        for Fi in range(1, M + 1):
            idx = Fi - 1
            if Fi == 1:
                F_P[:, idx:idx+1] = k_B1
            elif Fi >= 2 and Fi < M-D+1:
                F_P[:, idx:idx+1] = k_B2
            elif Fi == M-D+1:
                F_P[:, idx:idx+1] = k_Bmd1
            elif Fi >= M-D+1 and Fi < M-1:
                F_P[:, idx:idx+1] = k_Y2
            elif Fi > M-D+1 and Fi == M-1:
                F_P[:, idx:idx+1] = k_Yd1
            elif Fi == M:
                F_P[:, idx:idx+1] = k_Yd

        # Replicate Legacy logic (calc_F_D) for Set D
        for Fi in range(1, M + 1):
            idx = Fi - 1
            if Fi <= M-D+1:
                F_D[:, idx:idx+1] = k_Y1
            elif Fi > M-D+1 and Fi <= M-2:
                F_D[:, idx:idx+1] = k_Y2
            elif Fi > M-D+1 and Fi == M-1:
                F_D[:, idx:idx+1] = k_Yd1
            elif Fi == M:
                F_D[:, idx:idx+1] = k_Yd

        # Combine based on X[:, 0] > 1/3
        condition = (X[:, 0:1] > 1/3)
        F = np.where(condition, F_D, F_P)
        
        # Apply (1+g) scaling to ALL
        F = F * (1 + g)
        
        return {'F': F}

    def get_K(self):
        return self.K

    def ps(self, n_points: int = 100):
        """Analytical sampling of DPF5 Pareto Set (Requires xM = x1)."""
        M = self.M
        N = self.N
        res = np.zeros((n_points, N))
        # Variables x1..xM-1 are position-like but x1 also drives g
        # Standard ps for DPF5: x_i in [0, 1] for i < M, x_M = x_1, x_i = 0.5 for i > M
        # Wait, indexing: x_1 is index 0. x_M is index M-1.
        X_pos = np.random.random((n_points, M - 1))
        res[:, :M-1] = X_pos
        res[:, M-1] = X_pos[:, 0] # x_M = x_1
        if N > M:
            res[:, M:] = 0.5
        return res
