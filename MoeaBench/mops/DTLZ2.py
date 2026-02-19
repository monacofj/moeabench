# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_mop import BaseMop

class DTLZ2(BaseMop):
    """
    DTLZ2 benchmark problem.
    """
    def __init__(self, **kwargs):
        self.K = kwargs.pop('K', 10)
        m_val = kwargs.get('M', 3)
        if 'N' not in kwargs:
            kwargs['N'] = m_val + self.K - 1
        super().__init__(**kwargs)

    def validate(self):
        """
        DTLZ2 requires M-1 variables for position on the manifold 
        and at least 1 variable (K) for the distance function g.
        Reference: Deb et al. (2002) 'Scalable multi-objective optimization test problems'.
        """
        super().validate()
        if self.N < self.M:
            raise ValueError(
                f"DTLZ2 requires N >= M variables to maintain its mathematical structure.\n"
                f"M-1 variables are needed for position on the (M-1)-dimensional manifold, "
                f"and at least 1 variable is required for the distance function g (provided N={self.N}, M={self.M})."
            )

    def evaluation(self, X, n_ieq_constr=0):
        """
        Standard DTLZ2 evaluation.
        """
        X = np.atleast_2d(X)
        M = self.M
        
        # g = sum ( (xi - 0.5)^2 ) for i = M to N
        X_m = X[:, M-1:]
        g = np.sum((X_m - 0.5)**2, axis=1).reshape(-1, 1)
        
        return self._spherical_evaluation(X, g)

    def _spherical_evaluation(self, X, g, theta=None):
        M = self.M
        F = np.zeros((X.shape[0], M))
        
        if theta is None:
            # Standard DTLZ2-4 theta
            theta = X[:, :M-1] * (np.pi / 2)
        elif isinstance(theta, list):
            # DTLZ5-6 return list of columns
            theta = np.column_stack(theta)
            
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        for i in range(M):
            f = (1 + g).flatten()
            if i < M - 1:
                f *= np.prod(cos_theta[:, :M-i-1], axis=1)
            
            if i > 0:
                f *= sin_theta[:, M-i-1]
                
            F[:, i] = f
        return {'F': F}

    def ps(self, n_points=100):
        """Analytical sampling of DTLZ2 Pareto Set (Deterministic)."""
        M = self.M
        N = self.N
        res = np.zeros((n_points, N))
        
        # Deterministic sampling for position on the manifold
        if M == 2:
            res[:, 0] = np.linspace(0, 1, n_points)
        else:
            # Simple quasi-random grid for M > 2
            # For verification and standards, a fixed Sobol/Halton or even 
            # a seeded random is better than raw random.
            rng = np.random.RandomState(42) # Scientific reproducibility seed
            res[:, :M-1] = rng.random((n_points, M - 1))
            
        res[:, M-1:] = 0.5
        return res

    def get_K(self):
        return self.K