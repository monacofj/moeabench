# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from itertools import accumulate, repeat, cycle, islice
from .base_mop import BaseMop

class BaseDPF(BaseMop):
    """
    Shared logic for Degenerate Pareto Front (DPF) benchmarks.
    
    References:
        Zhen, L., Li, M., Cheng, R., Peng, D., & Yao, X. (2018). "Multiobjective Test 
        Problems with Degenerate Pareto Fronts." IEEE Trans. Evol. Comput.
    """
    def __init__(self, **kwargs):
        d_val = kwargs.pop('D', 2)
        m_val = kwargs.get('M', 3)
        if d_val < 2 or d_val >= m_val:
            d_val = 2
        self.D = d_val
        
        self.K = kwargs.pop('K', 5)
        
        # Calculate N if not provided
        if 'N' not in kwargs:
            kwargs['N'] = self.D + self.K - 1
            
        super().__init__(**kwargs)
        # Static chaos pool: (M-D) projection columns, each needing weights if DPF1/2
        # For DPF1/2: N_weights = (M-D) * D
        # For DPF3/4: N_weights = (M-D)
        # We generate a large enough pool to cover all cases.
        self._chaos_pool = self._generate_chaos(max((self.M - self.D) * self.D, self.M - self.D))

    def validate(self):
        """
        DPF specific validation: ensures that the projection 
        from D to M objectives is mathematically possible given N.
        """
        super().validate()
        # N = D + K - 1.
        # Removing legacy M <= N constraint to support any number of objectives
        # as per the original degenerate front design.

    def _generate_chaos(self, size):
        """Generates a static chaotic sequence using the Logistic Map (x0=0.1, r=3.8)."""
        res = []
        x = 0.1
        for _ in range(size):
            x = 3.8 * x * (1 - x)
            res.append(x)
        return np.array(res)

    def _project(self, F_base, square=False):
        """Standard DPF1/2 projection logic using static chaos."""
        if self.M == self.D:
            return F_base
        
        M_D = self.M - self.D
        D = self.D
        redundant = []
        
        # Reshape chaos pool for DPF1/2: (M-D) columns of weights (each of size D)
        weights_matrix = self._chaos_pool[:M_D * D].reshape(M_D, D)
        
        for i in range(M_D):
            w = weights_matrix[i, :]
            proj = np.dot(F_base, w)
            if square:
                proj = proj**2
            redundant.append(proj)
            
        return np.concatenate((F_base, np.column_stack(redundant)), axis=1)

    def ps(self, n_points=100):
        """Analytical sampling of DPF Pareto Set."""
        D = self.D
        N = self.N
        res = np.zeros((n_points, N))
        res[:, :D-1] = np.random.random((n_points, D - 1))
        res[:, D-1:] = 0.5
        return res

    def get_D(self):
        return self.D

    def get_K(self):
        return self.K
