# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Published DPF3 objective evaluation (Zhen et al., equations 15–16).

The chaotic projection thresholds are *sorted for DPF3 only*. Other DPF
benchmarks retain their own BaseDPF chaotic-sequence conventions.

This change corrects objective evaluation; angular PS sampling, PF
discretization and persisted GT migration are separate review steps.
"""

import numpy as np

from .base_dpf import BaseDPF


class DPF3(BaseDPF):
    """DPF3: inverted hyperspherical essentials and ordered projections."""

    def eval_base_functions(self, X, g, D):
        """Evaluate (1 + g) * (1 - h(theta)) without 1-cos cancellation."""
        theta = np.power(X[:, :D-1], 100) * (np.pi / 2)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 1 - prod(cos(theta_j)) =
        #   sum_j (prod_{k<j} cos(theta_k)) * 2*sin(theta_j/2)^2.
        # Every term is nonnegative for feasible theta in [0, pi/2].
        def one_minus_cos_product(end):
            complement = np.zeros(X.shape[0], dtype=float)
            prefix = np.ones(X.shape[0], dtype=float)
            for j in range(end):
                complement += prefix * (2 * np.sin(theta[:, j] / 2)**2)
                prefix *= cos_theta[:, j]
            return complement

        Y = np.empty((X.shape[0], D), dtype=float)
        scale = np.asarray(g).reshape(-1) + 1.0
        for i in range(D):
            # Standard D-dimensional spherical coordinate h_i.
            cos_count = D - i - 1
            complement = one_minus_cos_product(cos_count)
            if i > 0:
                last = theta[:, cos_count]
                # 1 - sin(last), stable when last approaches pi/2.
                one_minus_sin = 2 * np.sin((np.pi / 2 - last) / 2)**2
                complement = one_minus_sin + sin_theta[:, cos_count] * complement
            Y[:, i] = complement * scale
        return Y

    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        D, M = self.D, self.M
        g = np.sum((X[:, D-1:] - 0.5)**2, axis=1).reshape(-1, 1)
        Y = self.eval_base_functions(X, g, D)

        if M == D:
            return {"F": Y}

        # The published DPF3 min/max projections require *ascending* eta.
        # Sorting is deliberately confined to DPF3, not shared BaseDPF.
        U = M - D
        eta = np.sort(self._chaos_pool[:U])
        Yd = Y[:, D-1:D]

        redundant = [np.minimum(Yd, eta[0])]
        for i in range(U - 1):
            redundant.append(np.minimum(np.maximum(Yd, eta[i]), eta[i+1]))
        redundant.append(np.maximum(Yd, eta[-1]))

        return {"F": np.concatenate((Y[:, :D-1], np.column_stack(redundant)), axis=1)}

    def get_K(self):
        return self.K

    def pf(self, n_points: int = 100):
        """Legacy finite PS sampling/filtering retained for separate migration.

        Correcting the published evaluator does not itself certify the
        existing finite-reference algorithm. The historical method is
        preserved temporarily in this draft until its MOP-specific angular
        sampler and any affected regression contracts are reviewed.
        """
        F = super().pf(n_points)
        try:
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

            fronts = NonDominatedSorting().do(F)
            return F[fronts[0]]
        except ImportError:
            return F
