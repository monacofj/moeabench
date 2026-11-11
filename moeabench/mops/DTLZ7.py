# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_mop import BaseMop
from ._dtlz7_pareto_regions import sample_dtlz7_record_parameters

class DTLZ7(BaseMop):
    """
    DTLZ7 benchmark problem. 
    Disconnected Pareto front.
    """
    def __init__(self, **kwargs):
        self.K = kwargs.pop('K', 20)
        m_val = kwargs.get('M', 3)
        if 'N' not in kwargs:
            kwargs['N'] = m_val + self.K - 1
        super().__init__(**kwargs)

    def validate(self):
        """
        DTLZ7 requires M-1 variables for position on the manifold 
        and at least 1 variable (K) for the distance function g.
        Reference: Deb et al. (2002) 'Scalable multi-objective optimization test problems'.
        """
        super().validate()
        if self.N < self.M:
            raise ValueError(
                f"DTLZ7 requires N >= M variables to maintain its mathematical structure.\n"
                f"M-1 variables are needed for position on the (M-1)-dimensional manifold, "
                f"and at least 1 variable is required for the distance function g (provided N={self.N}, M={self.M})."
            )

    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        M = self.M
        K = self.K
        
        # g = 1 + 9/K * sum(Xi) for i=M to N
        X_m = X[:, M-1:]
        g = 1 + 9 / K * np.sum(X_m, axis=1).reshape(-1, 1)
        
        F = np.zeros((X.shape[0], M))
        # f1..fm-1 = x1..xm-1
        F[:, :M-1] = X[:, :M-1]
        
        # h = M - sum ( fi/(1+g) * (1+sin(3*pi*fi)) )
        h = M - np.sum( (F[:, :M-1] / (1 + g)) * (1 + np.sin(3 * np.pi * F[:, :M-1])), axis=1).reshape(-1, 1)
        
        F[:, M-1:] = (1 + g) * h
        return {'F': F}

    def ps(self, n_points=100):
        """Sample only the analytically Pareto-optimal disconnected regions.

        Parameter-uniform on the exact record intervals, not objective-area
        uniform. Never treat this as a recalibration of frozen legacy GT.
        """
        M = self.M
        N = self.N
        res = np.zeros((n_points, N))
        # The unrestricted [0,1]^(M-1) parameter cube contains many
        # dominated points: DTLZ7 has 2^(M-1) disconnected PF components.
        # Sample instead from the exact Pareto-record domain of
        # x*(1+sin(3*pi*x)), retaining the historical private RNG seed.
        res[:, :M-1] = sample_dtlz7_record_parameters(
            n_points, M-1, seed=42
        )
        # Optimal g=1 when xi=0 for i >= M
        res[:, M-1:] = 0.0
        return res

    def get_K(self):
        return self.K
