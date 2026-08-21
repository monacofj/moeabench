# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Monte Carlo raw-coordinate evaluator for Ordinal Hypervolume."""

import numpy as np
from moocore import hv_approx


class GEN_mc_ordinal_hypervolume:
    """Evaluate already transformed ordinal fronts with common-random sampling."""

    def __init__(self, hist_F, ref_point, n_samples=100000, seed=None,
                 progress_callback=None, **kwargs):
        if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        if seed is not None and (not isinstance(seed, (int, np.integer)) or isinstance(seed, bool)):
            raise ValueError("mc_seed must be None or an integer.")
        self.hist_F = hist_F
        self.ref_point = np.asarray(ref_point, dtype=float)
        self.n_samples = int(n_samples)
        self.seed = None if seed is None else int(seed)
        self.progress_callback = progress_callback

    def evaluate(self):
        results = []
        for front in self.hist_F:
            if len(front) == 0:
                value = 0.0
            else:
                value = float(hv_approx(
                    front,
                    self.ref_point,
                    nsamples=self.n_samples,
                    seed=self.seed,
                    method="DZ2019-MC",
                ))
            results.append(value)
            if self.progress_callback is not None:
                self.progress_callback()
        return np.asarray(results, dtype=float)
