# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .base_mop import BaseMop


_ANALYTICAL_SAMPLE_SEED = 42


def _sample_positive_hypersphere(n_points, n_objectives):
    """Return deterministic samples approximately uniform on the positive unit sphere."""
    if n_objectives == 2:
        theta = np.linspace(0.0, np.pi / 2.0, n_points)
        return np.column_stack((np.cos(theta), np.sin(theta)))

    rng = np.random.RandomState(_ANALYTICAL_SAMPLE_SEED)
    directions = np.abs(rng.standard_normal((n_points, n_objectives)))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)

    # A zero Gaussian vector has probability zero in exact arithmetic, but keep
    # the normalization robust in case a pathological RNG result ever occurs.
    zero = norms[:, 0] == 0.0
    while np.any(zero):
        directions[zero] = np.abs(rng.standard_normal((np.count_nonzero(zero), n_objectives)))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        zero = norms[:, 0] == 0.0

    directions /= norms

    # Preserve the exact analytical extrema whenever the requested sample is
    # large enough to contain every objective axis. This keeps ideal/nadir
    # bounds exact while perturbing the surface distribution only negligibly.
    if n_points >= n_objectives:
        directions[:n_objectives] = np.eye(n_objectives)

    return directions


def _hypersphere_directions_to_angles(directions):
    """Invert the DTLZ2 spherical map from objective directions to theta angles."""
    directions = np.asarray(directions, dtype=float)
    n_points, n_objectives = directions.shape
    theta = np.zeros((n_points, n_objectives - 1), dtype=float)

    for j in range(n_objectives - 1):
        objective = n_objectives - j - 1
        radial = np.linalg.norm(directions[:, :objective], axis=1)
        theta[:, j] = np.arctan2(directions[:, objective], radial)

    return theta


class DTLZ2(BaseMop):
    """
    DTLZ2 benchmark problem.
    """
    def __init__(self, **kwargs):
        self.K = kwargs.pop('K', 10)
        m_val = kwargs.get('M', 3)
        if 'N' not in kwargs:
            kwargs['N'] = m_val + self.K - 1
        super().__init__(**kwargs)

    def validate(self):
        """
        DTLZ2 requires M-1 variables for position on the manifold
        and at least 1 variable (K) for the distance function g.
        Reference: Deb et al. (2002) 'Scalable multi-objective optimization test problems'.
        """
        super().validate()
        if self.N < self.M:
            raise ValueError(
                f"DTLZ2 requires N >= M variables to maintain its mathematical structure.\n"
                f"M-1 variables are needed for position on the (M-1)-dimensional manifold, "
                f"and at least 1 variable is required for the distance function g (provided N={self.N}, M={self.M})."
            )

    def evaluation(self, X, n_ieq_constr=0):
        """
        Standard DTLZ2 evaluation.
        """
        X = np.atleast_2d(X)
        M = self.M

        # g = sum ( (xi - 0.5)^2 ) for i = M to N
        X_m = X[:, M-1:]
        g = np.sum((X_m - 0.5)**2, axis=1).reshape(-1, 1)

        return self._spherical_evaluation(X, g)

    def _spherical_evaluation(self, X, g, theta=None):
        M = self.M
        F = np.zeros((X.shape[0], M))

        if theta is None:
            # Standard DTLZ2-4 theta
            theta = X[:, :M-1] * (np.pi / 2)
        elif isinstance(theta, list):
            # DTLZ5-6 return list of columns
            theta = np.column_stack(theta)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        for i in range(M):
            f = (1 + g).flatten()
            if i < M - 1:
                f *= np.prod(cos_theta[:, :M-i-1], axis=1)

            if i > 0:
                f *= sin_theta[:, M-i-1]

            F[:, i] = f
        return {'F': F}

    def ps(self, n_points=100):
        """Analytical sampling of the DTLZ2 Pareto Set (deterministic)."""
        M = self.M
        N = self.N
        res = np.zeros((n_points, N))

        if M == 2:
            # Uniform theta is uniform arc length in the positive circle quadrant.
            res[:, 0] = np.linspace(0, 1, n_points)
        else:
            directions = _sample_positive_hypersphere(n_points, M)
            theta = _hypersphere_directions_to_angles(directions)
            res[:, :M-1] = theta / (np.pi / 2)

        res[:, M-1:] = 0.5
        return res

    def get_K(self):
        return self.K
