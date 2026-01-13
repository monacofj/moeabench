# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .DTLZ2 import DTLZ2

class DTLZ4(DTLZ2):
    """
    DTLZ4 benchmark problem.
    Same as DTLZ2 but with biased sampling of the search space.
    """
    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        M = self.M
        X_m = X[:, M-1:]
        g = np.sum((X_m - 0.5)**2, axis=1).reshape(-1, 1)
        
        # Biased theta: x_i^100
        theta = (X[:, :M-1]**100) * (np.pi / 2)
        
        return self._spherical_evaluation(X, g, theta=theta)