# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
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
        
        # Construct Set P (Length M)
        # Legacy calc_F_P:
        # Fi=1 (0): B1
        # 2 <= Fi < M-D+1: B2
        # Fi == M-D+1: Bmd1
        # M-D+1 < Fi < M-1: Y2
        # Fi == M-1: Yd1
        # Fi == M: Yd
        
        # Vectorized Set P
        F_P = np.zeros((X.shape[0], M))
        # idx 0: B1
        F_P[:, 0:1] = k_B1
        # idx 1 to M-D-1 (inclusive start, exclusive end logic?):
        # Fi goes 2 to M-D. Count = M-D - 2 + 1 = M-D-1.
        if M-D > 1:
            F_P[:, 1:M-D] = k_B2
        # idx M-D: Bmd1 (Fi = M-D+1 -> idx M-D)
        if M-D < M:
            F_P[:, M-D:M-D+1] = k_Bmd1
        # idx M-D+1 to M-2: Y2
        if M-2 > M-D:
            F_P[:, M-D+1:M-1] = k_Y2
        # idx M-2: Yd1
        if M-1 > 0:
            F_P[:, M-2:M-1] = k_Yd1
        # idx M-1: Yd
        F_P[:, M-1:M] = k_Yd


        # Construct Set D (Length M)
        # Legacy calc_F_D:
        # Fi <= M-D+1 (idx 0 to M-D): Y1
        # M-D+1 < Fi <= M-2 (idx M-D+1 to M-3): Y2 ??
        # Wait: "Fi > M-D+1 and Fi <= M-2"
        # idx M-D+1 to M-2 (indices).
        # Fi == M-1 (idx M-2): Yd1 (Legacy calls it "R2(2)" which is ... wait)
        # R2 set: [Y1, Y2, Yd1, Yd] (indices 0,1,2,3 mapped to Y1, Y2, Yd1, Yd)
        # Correct.
        # Fi == M (idx M-1): Yd
        
        F_D = np.zeros((X.shape[0], M))
        # 0 to M-D: Y1
        F_D[:, :M-D+1] = k_Y1
        # M-D+1 to M-2: Y2
        if M-2 > M-D:
            F_D[:, M-D+1:M-1] = k_Y2
        # M-2: Yd1
        if M-1 > 0:
            F_D[:, M-2:M-1] = k_Yd1
        # M-1: Yd
        F_D[:, M-1:M] = k_Yd

        # Combine based on X[:, 0] > 1/3
        # If X0 > 1/3: Set D. Else: Set P.
        condition = (X[:, 0:1] > 1/3)
        F = np.where(condition, F_D, F_P)
        
        # Apply (1+g) scaling to ALL
        F = F * (1 + g)
        
        return {'F': F}

    def get_K(self):
        return self.K
