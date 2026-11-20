# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Published DPF5 evaluation with the full P and degenerate D branches.

This corrects the mathematical objective evaluator without changing the
finite PS sampler or historical GT/calibration policy. Those require
separate geometry-specific validation before a production merge.
"""

import numpy as np

from .base_dpf import BaseDPF


class DPF5(BaseDPF):
    """DPF5: P=(M-1)-dimensional, D=(D-1)-dimensional angular fronts."""

    def __init__(self, **kwargs):
        if 'N' not in kwargs:
            M = kwargs.get('M', 3)
            D = kwargs.get('D', 2)
            if D < 2 or D >= M:
                D = 2
            K = kwargs.get('K', 5)
            kwargs['N'] = max(D + K - 1, M)
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        self._require_supported_dimensions()

    def _require_supported_dimensions(self):
        """The published distance function uses x_M explicitly."""
        if self.N < self.M:
            raise ValueError(
                f"DPF5 requires N >= M because its distance function uses x_M "
                f"(got M={self.M}, N={self.N}). Provide N >= {self.M}. "
                "Other DPF benchmarks can allow M > N."
            )

    @staticmethod
    def _unit_hypersphere(theta):
        """Nonnegative, unit-norm direction with theta.shape[1]+1 objectives.

        Coordinates follow the standard DTLZ2 hyperspherical convention.
        The implementation is vectorized over individuals and loops only
        over the number of objective coordinates.
        """
        n, q = theta.shape
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        direction = np.empty((n, q + 1), dtype=float)

        for j in range(q + 1):
            cos_count = q - j
            coordinate = np.prod(cos_theta[:, :cos_count], axis=1)
            if j > 0:
                coordinate = coordinate * sin_theta[:, cos_count]
            direction[:, j] = coordinate
        return direction

    def evaluation(self, X, n_ieq_constr=0):
        self._require_supported_dimensions()
        X = np.atleast_2d(X)
        M, D = self.M, self.D

        g = (X[:, M-1] - X[:, 0])**2
        g += np.sum((X[:, M:] - 0.5)**2, axis=1)

        theta = X[:, :M-1] * (np.pi / 2)
        P = self._unit_hypersphere(theta)

        # The D direction repeats its first essential spherical component
        # M-D+1 times, scaled so the resulting M-vector retains unit norm.
        essential = self._unit_hypersphere(theta[:, :D-1])
        repeats = M - D + 1
        D_front = np.concatenate(
            (np.repeat(essential[:, :1] / np.sqrt(repeats), repeats, axis=1),
             essential[:, 1:]),
            axis=1,
        )

        # Per the publication: x1 == 1/3 belongs to D, NOT P.
        choose_D = X[:, 0] >= (1 / 3)
        direction = np.where(choose_D[:, None], D_front, P)
        return {"F": direction * (1 + g[:, None])}

    def get_K(self):
        return self.K

    def ps(self, n_points: int = 100):
        """Legacy finite PS sampling, retained pending branchwise GT review.

        Correct objective evaluation does not certify this sampler's point
        budget or coverage of the mixed-dimensional continuous PF.
        """
        self._require_supported_dimensions()
        M, N = self.M, self.N
        res = np.zeros((n_points, N))
        rng = np.random.RandomState(42)

        if M == 3:
            side = int(np.sqrt(n_points))
            grid_val = np.linspace(0, 1, side)
            xv, yv = np.meshgrid(grid_val, grid_val)
            X_pos = np.column_stack([xv.ravel(), yv.ravel()])
            if len(X_pos) != n_points:
                res = np.zeros((len(X_pos), N))
        else:
            X_pos = rng.random((n_points, M - 1))

        res[:, :M-1] = X_pos
        res[:, M-1] = res[:, 0]
        if N > M:
            res[:, M:] = 0.5
        return res
