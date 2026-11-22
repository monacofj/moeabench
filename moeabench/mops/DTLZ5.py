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

        The historical g=0 sampling is valid as a parametrized degenerate
        Pareto subset, but it is not the *complete* front when M >= 4:
        nonzero-g points can also be Pareto optimal. Preserve the historical
        sequence for regression reproducibility and warn callers explicitly.
        Until a complete front is mathematically validated, do not treat
        these samples as exhaustive Ground Truth for IGD/IGD+ or HV.
        """
        M = self.M
        N = self.N
        if M >= 4:
            raise NotImplementedError(
                "DTLZ5(M>=4): the original Pareto front includes "
                "Pareto-optimal solutions with g>0. The historical g=0 "
                "sampler is incomplete; ps()/pf() cannot provide a "
                "scientifically valid full Ground Truth until issue #65 "
                "is resolved."
            )
        res = np.zeros((n_points, N))

        if M == 2:
            res[:, 0] = np.linspace(0, 1, n_points)
        else:
            rng = np.random.RandomState(42)
            res[:, :M-1] = rng.random((n_points, M - 1))

        res[:, M-1:] = 0.5
        return res
