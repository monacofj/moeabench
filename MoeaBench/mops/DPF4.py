# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
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
        
        # g = sum(xi - 0.5)^2 for variables D-1 onwards (Standard)
        g = 100 * (K + np.sum((X[:, D-1:] - 0.5)**2 - np.cos(20 * np.pi * (X[:, D-1:] - 0.5)), axis=1)).reshape(-1, 1)
        # Wait, I need to check K_DPF4 g definition (Step 2568)
        # 66: 100*(K+np.sum(((Xi-0.5)**2)-np.cos(20*np.pi*(Xi-0.5))))
        # Yes, it's Rastrigin-like DTLZ1/3 style g.
        
        # Calculate base objectives Y (D objectives)
        Y = self.eval_base_functions(X, g, D)
        
        Yd = Y[:, D-1:D] # The last base objective
        
        if M == D:
            return {'F': Y}
            
        # Dynamic chaos generation (M-D columns per sample)
        vet_chaos = self._calc_dynamic_chaos(X.shape[0]) # (Samples, M-D)
        
        redundant = []
        for row in range(X.shape[0]):
            row_vals = []
            yd_val = Yd[row, 0]
            chaos_row = vet_chaos[row]
            
            # FD: min(Yd, chaos[0])**2
            row_vals.append(min(yd_val, chaos_row[0])**2)
            
            # FD1: min(max(Yd, C[i]), C[i+1])**2
            for col in range(0, M-D-1):
                val = min(max(yd_val, chaos_row[col]), chaos_row[col+1])**2
                row_vals.append(val)
            
            # FM: max(Yd, C[last])**2
            row_vals.append(max(yd_val, chaos_row[M-D-1])**2)
            
            redundant.append(row_vals)
            
        F_redundant = np.array(redundant)
        
        # Combine: Y[:, :D-1] + F_redundant
        # Note: Base objectives Y1..Yd-1 are NOT squared in eval_base.
        # Only the redundant projections are squared.
        F = np.concatenate((Y[:, :D-1], F_redundant), axis=1)
        
        return {'F': F}

    def get_K(self):
        return self.K