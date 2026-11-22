# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .DTLZ2 import DTLZ2


class DTLZ5(DTLZ2):
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

        For M>=4, sample the original full-front geometry by globally
        bounded scalarization, including g>0 Pareto regions. M<=3 retains
        the analytically valid historical degenerate sampler.
        """
        M = self.M
        N = self.N
        if M >= 4:
            from ._dtlz56_front import sample_original_dtlz56_ps
            return sample_original_dtlz56_ps(self, n_points)
        res = np.zeros((n_points, N))

        if M == 2:
            res[:, 0] = np.linspace(0, 1, n_points)
        else:
            rng = np.random.RandomState(42)
            res[:, :M-1] = rng.random((n_points, M - 1))

        res[:, M-1:] = 0.5
        return res
