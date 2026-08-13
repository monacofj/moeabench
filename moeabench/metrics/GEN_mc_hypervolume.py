# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from moocore import hv_approx


class GEN_mc_hypervolume:
    """
    Monte Carlo approximation of Hypervolume using moocore's native backend.
    Useful for many-objective optimization (M > 8).
    """
    def __init__(self, hist_F, M, approx_ideal, approx_nadir, n_samples=100000,
                 seed=None, progress_callback=None, **kwargs):
        if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        self.hist_F = hist_F
        self.M = M
        self.ideal = approx_ideal
        self.nadir = approx_nadir
        self.n_samples = int(n_samples)
        self.seed = seed
        self.progress_callback = progress_callback

    def evaluate(self):
        # Fixed normalized reference point, consistent with GEN_hypervolume.
        ref_point = np.full(self.M, 1.1)

        results = []
        for F in self.hist_F:
            if len(F) == 0:
                results.append(0.0)
                if self.progress_callback is not None:
                    self.progress_callback()
                continue
            
            # Normalize raw objectives to the fixed ideal/nadir context.
            F_norm = (F - self.ideal) / (self.nadir - self.ideal + 1e-10)

            # Reusing the same integer seed for each generation makes the native
            # Monte Carlo backend use common random weights. This preserves
            # reproducibility and reduces noise in temporal comparisons.
            value = hv_approx(
                F_norm,
                ref_point,
                nsamples=self.n_samples,
                seed=self.seed,
                method="DZ2019-MC",
            )
            results.append(float(value))

            if self.progress_callback is not None:
                self.progress_callback()

        return np.array(results)
