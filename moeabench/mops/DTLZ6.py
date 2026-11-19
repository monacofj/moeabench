# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import warnings
from .DTLZ5 import DTLZ5

class DTLZ6(DTLZ5):
    """
    DTLZ6 benchmark problem.
    Biased degenerate spherical problem.
    """
    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        M = self.M
        X_m = X[:, M-1:]
        
        # g = sum(xi^0.1)
        g = np.sum(X_m**0.1, axis=1).reshape(-1, 1)
        
        # Theta logic same as DTLZ5
        theta = [X[:, 0:1] * np.pi/2]
        for i in range(1, M-1):
            theta.append((np.pi / (4 * (1 + g))) * (1 + 2 * g * X[:, i:i+1]))
            
        return self._spherical_evaluation(X, g, theta=theta)
            
    def ps(self, n_points=100):
        """Sample DTLZ6's historical g=0 Pareto subset.

        For M>=4, nonzero-g Pareto optima are omitted. The output is
        not a complete GT; retain its deterministic legacy coordinates.
        """
        M = self.M
        N = self.N
        if M >= 4:
            warnings.warn(
                "DTLZ6(M>=4): ps()/pf() samples only the g=0 subset; "
                "the true Pareto front also contains nonzero-g points. "
                "This sample is not a complete Ground Truth for "
                "IGD/IGD+ or absolute HV.",
                UserWarning,
                stacklevel=2,
            )
        res = np.zeros((n_points, N))
        rng = np.random.RandomState(42)
        res[:, :M-1] = rng.random((n_points, M - 1))
        # Optimal g=0 when xi=0 for i >= M
        res[:, M-1:] = 0.0
        return res
