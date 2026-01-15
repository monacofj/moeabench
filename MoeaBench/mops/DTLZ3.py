# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .DTLZ2 import DTLZ2

class DTLZ3(DTLZ2):
    """
    DTLZ3 benchmark problem.
    Same as DTLZ2 but with a highly multimodal g function.
    """
    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        M = self.M
        
        # g = 100 * (K + sum((xi - 0.5)**2 - cos(20 * pi * (xi - 0.5))))
        X_m = X[:, M-1:]
        g = 100 * (self.K + np.sum((X_m - 0.5)**2 - np.cos(20 * np.pi * (X_m - 0.5)), axis=1)).reshape(-1, 1)
        
        return self._spherical_evaluation(X, g)