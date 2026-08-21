# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exact raw-coordinate evaluator for Ordinal Hypervolume."""

import numpy as np
from pymoo.indicators.hv import Hypervolume


class GEN_ordinal_hypervolume:
    """Evaluate already transformed ordinal fronts without normalization."""

    def __init__(self, hist_F, ref_point, progress_callback=None, **kwargs):
        self.hist_F = hist_F
        self.ref_point = np.asarray(ref_point, dtype=float)
        self.progress_callback = progress_callback
        super().__init__(**kwargs)

    def evaluate(self):
        metric = Hypervolume(
            ref_point=self.ref_point,
            norm_ref_point=False,
            zero_to_one=False,
        )
        results = []
        for front in self.hist_F:
            value = 0.0 if len(front) == 0 else float(metric.do(front))
            results.append(value)
            if self.progress_callback is not None:
                self.progress_callback()
        return np.asarray(results, dtype=float)
