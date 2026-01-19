# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_dpf import BaseDPF

class DPF4(BaseDPF):
    """
    DPF4 benchmark problem.
    Similar to DPF3 but with:
    1. Linear theta mapping (usually, legacy calls calc_TH(X) without alpha=100)
    2. Squared projection functions (FD**2, FM**2)
    """
    def eval_base_functions(self, X, g, D):
        # Legacy K_DPF4 calls calc_TH(X) without argument.
        # Assuming default behavior is linear: x * pi/2
        theta = X[:, :D-1] * (np.pi / 2)
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        Y = np.zeros((X.shape[0], D))
        
        for i in range(D):
            f_norm = 1.0
            if i < D - 1:
                f_norm *= np.prod(cos_theta[:, :D-i-1], axis=1)
            
            if i > 0:
                f_norm *= sin_theta[:, D-i-1]
                
            Y[:, i] = (1 - f_norm) * (1 + g).flatten()
            
        return Y

    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        D, K, M = self.D, self.K, self.M
        
        # g = Rastrigin-like DTLZ1/3 style g
        g = 100 * (K + np.sum((X[:, D-1:] - 0.5)**2 - np.cos(20 * np.pi * (X[:, D-1:] - 0.5)), axis=1)).reshape(-1, 1)
        
        # Calculate base objectives Y (D objectives)
        Y = self.eval_base_functions(X, g, D)
        
        Yd = Y[:, D-1:D] # The last base objective
        
        if M == D:
            return {'F': Y}
            
        # Use static chaos pool from BaseDPF
        U = M - D
        chaos = self._chaos_pool[:U]
        
        redundant = []
        # FD (first redundant): min(Yd, chaos[0])**2
        redundant.append(np.minimum(Yd, chaos[0])**2)
        
        # FD1 (middle): min(max(Yd, chaos[i]), chaos[i+1])**2
        for i in range(U - 1):
            val = np.minimum(np.maximum(Yd, chaos[i]), chaos[i+1])**2
            redundant.append(val)
        
        # FM (last redundant): max(Yd, chaos[U-1])**2
        redundant.append(np.maximum(Yd, chaos[U-1])**2)
        
        F_redundant = np.column_stack(redundant)
        
        # Combine: Y[:, :D-1] + F_redundant
        F = np.concatenate((Y[:, :D-1], F_redundant), axis=1)
        
        return {'F': F}

    def get_K(self):
        return self.K