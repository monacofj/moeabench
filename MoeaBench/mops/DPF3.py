# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_dpf import BaseDPF

class DPF3(BaseDPF):
    """
    DPF3 benchmark problem.
    High-dimensional problem with chaos-based min/max projection (non-linear).
    Base problem is an inverted DTLZ-like form.
    """
    def eval_base_functions(self, X, g, D):
        # Legacy K_DPF3 Uses calc_TH(X, 100) typical of DTLZ4
        # Assumed calc_TH(X, 100) -> X**100 * pi/2
        theta = (X[:, :D-1]**100) * (np.pi / 2)
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Y1..Yd1 (first D-1 objectives)
        # Y1 = (1 - prod(cos(all))) * (1+g)
        # Y2 = (1 - prod(cos(all except last)) * sin(last)) * (1+g)
        # ...
        # Yd1 = (1 - cos(0)*sin(1)) * (1+g)  <-- Wait, legacy Yd1 checks indices carefully
        
        # Replicating logic from K_DPF3
        # Y1: 1 - prod(cos(theta[:D-1]))
        # Y2: 1 - prod(cos(theta[:D-2])) * sin(theta[D-2])
        # Yd1: 1 - cos(theta[0]) * sin(theta[1])  (if D > 2?)
        
        # Actually, let's generalize:
        # Base functions are 1 - (Standard DTLZ2 normalized functions) ?
        # Standard DTLZ2 normalized: f = prod(cos..) * sin..
        # DPF3: 1 - f
        
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
        
        # g = sum(xi - 0.5)^2 for variables D-1 onwards
        g = np.sum((X[:, D-1:] - 0.5)**2, axis=1).reshape(-1, 1)
        
        # Calculate base objectives Y (D objectives)
        Y = self.eval_base_functions(X, g, D)
        
        Yd = Y[:, D-1:D] # The last base objective
        
        if M == D:
            return {'F': Y}
            
        # Use static chaos pool from BaseDPF
        U = M - D
        chaos = self._chaos_pool[:U]
        
        redundant = []
        # FD (first projection): min(Yd, chaos[0])
        redundant.append(np.minimum(Yd, chaos[0]))
        
        # FD1 (intermediates): min(max(Yd, chaos[i]), chaos[i+1])
        for i in range(U - 1):
            val = np.minimum(np.maximum(Yd, chaos[i]), chaos[i+1])
            redundant.append(val)
            
        # FM (last projection): max(Yd, chaos[U-1])
        redundant.append(np.maximum(Yd, chaos[U-1]))
            
        F_redundant = np.column_stack(redundant)
        
        # Build final objective matrix: D-1 base columns + M-D+1 projections = M columns
        F = np.concatenate((Y[:, :D-1], F_redundant), axis=1)
        
        return {'F': F}

    def get_K(self):
        return self.K