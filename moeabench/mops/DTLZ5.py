# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .DTLZ2 import DTLZ2


class DTLZ5(BaseMop if False else DTLZ2): # Inherit helper from DTLZ2
    """
    DTLZ5 benchmark problem.
    Degenerate spherical problem.
    """
    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        M = self.M
        X_m = X[:, M-1:]
        g = np.sum((X_m - 0.5)**2, axis=1).reshape(-1, 1)

        # Modified theta calculation for DTLZ5
        theta = [X[:, 0:1] * np.pi/2]
        for i in range(1, M-1):
            theta.append((np.pi / (4 * (1 + g))) * (1 + 2 * g * X[:, i:i+1]))

        return self._spherical_evaluation(X, g, theta=theta)

    def ps(self, n_points=100):
        """
        Analytical sampling of the degenerate DTLZ5 Pareto Set.

        Keep the historical sampling explicitly here instead of inheriting
        DTLZ2.ps(): DTLZ5 has a different, one-dimensional Pareto-front
        geometry for M > 2, so DTLZ2's hypersphere sampling is not applicable.
        """
        M = self.M
        N = self.N
        res = np.zeros((n_points, N))

        if M == 2:
            res[:, 0] = np.linspace(0, 1, n_points)
        else:
            rng = np.random.RandomState(42)
            res[:, :M-1] = rng.random((n_points, M - 1))

        res[:, M-1:] = 0.5
        return res
