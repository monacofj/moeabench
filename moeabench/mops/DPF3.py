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

    def ps(self, n_points: int = 100):
        """Generate the published g=0 Pareto set in *angular* coordinates.

        A maximal k^(D-1) endpoint-inclusive tensor grid fitting the
        requested budget is retained first. Its mathematical covering
        bound is sqrt(D-1)*pi/[4*(k-1)] for k >= 2 (real arithmetic).
        Degenerate boundary images are deduplicated in objective space and
        replaced with deterministic scrambled Sobol interior points.
        The returned PS thus has n_points *distinct float64 objectives*
        under the current published evaluator; a rigorous float64 error
        enclosure is a separate certification requirement.

        This DPF3-specific policy avoids decision-uniform x**100 bias and
        does not prescribe sampling for other MOPs.
        """
        from itertools import product
        from scipy.stats import qmc

        if not isinstance(n_points, (int, np.integer)) or n_points < 1:
            raise ValueError("n_points must be a positive integer")

        q = self.D - 1
        n = int(n_points)
        k = max(1, int(n ** (1 / q)))
        while (k + 1) ** q <= n:
            k += 1
        while k ** q > n:
            k -= 1

        # Tensor grid in ANGLES, not in decision-space x. Every exact
        # tensor-grid objective lies on the published global Pareto front.
        initial = np.asarray(list(product(np.linspace(0.0, 1.0, k), repeat=q)),
                             dtype=float).reshape(-1, q)

        chosen = []
        seen = set()

        def append_unique(normalized_angles):
            if len(chosen) >= n or len(normalized_angles) == 0:
                return
            decision = np.full((len(normalized_angles), self.N), 0.5)
            # x=(2*theta/pi)**(1/100); theta normalized by pi/2.
            decision[:, :q] = np.power(normalized_angles, 1.0 / 100.0)
            values = self.evaluation(decision)["F"]
            for row, obj in zip(decision, values):
                signature = tuple(float(v) for v in obj)
                if signature not in seen:
                    seen.add(signature)
                    chosen.append(row)
                    if len(chosen) == n:
                        break

        append_unique(initial)
        if len(chosen) < n:
            engine = qmc.Sobol(d=q, scramble=True, seed=11903)
            # Interior samples replenish numerical/analytical duplicates
            # without removing the support of the proven tensor grid.
            attempts = 0
            while len(chosen) < n and attempts < 5:
                needed = n - len(chosen)
                power = int(np.ceil(np.log2(max(2, needed * 2))))
                # Each attempt uses a fresh independent deterministic
                # subsequence. No global RNG state is affected.
                engine = qmc.Sobol(d=q, scramble=True, seed=11903 + attempts)
                append_unique(engine.random_base2(power))
                attempts += 1
            if len(chosen) != n:
                raise ValueError("Could not produce n_points distinct DPF3 objectives")

        return np.asarray(chosen, dtype=float)

    def pf(self, n_points: int = 100):
        """Evaluate the globally Pareto-optimal DPF3 angular PS sample.

        Nondominated sorting is unnecessary: mathematical membership was
        proved for every g=0 decision with correctly sorted thresholds.
        Sorting a finite sample cannot establish global membership and
        can silently shrink its requested cardinality.
        """
        return super().pf(n_points)
