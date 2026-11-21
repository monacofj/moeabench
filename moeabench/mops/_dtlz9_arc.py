# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Arc-length parameterization for the published DTLZ9 Pareto curve.

For minimization, f_1=...=f_(M-1)=cos(theta), f_M=sin(theta).
The objective-space arc-length differential is
sqrt(1 + (M-2) sin(theta)^2) dtheta. The incomplete elliptic
integral of the second kind gives its antiderivative.
"""
import numpy as np
from scipy.special import ellipeinc


def dtlz9_equal_arc_angles(m, n_points):
    """Deterministically invert exact elliptic arc length on [0, pi/2]."""
    if not isinstance(n_points, (int, np.integer)) or n_points < 1:
        raise ValueError("n_points must be a positive integer.")
    if m < 2:
        raise ValueError("DTLZ9 requires M>=2.")
    elliptic_parameter = 2 - m
    total = ellipeinc(np.pi / 2, elliptic_parameter)
    targets = np.linspace(0, total, n_points)
    theta = targets / total * (np.pi / 2)
    # Positive derivative throughout the domain; monotone, stable Newton.
    for _ in range(10):
        length = ellipeinc(theta, elliptic_parameter)
        speed = np.sqrt(1 + (m - 2) * np.sin(theta)**2)
        theta = np.clip(theta - (length - targets) / speed,
                        0, np.pi / 2)
    theta[0] = 0.0
    if n_points > 1:
        theta[-1] = np.pi / 2
    return theta


def dtlz9_equal_arc_objectives(m, n_points):
    theta = dtlz9_equal_arc_angles(m, n_points)
    objectives = np.empty((n_points, m), dtype=float)
    objectives[:, :-1] = np.cos(theta)[:, None]
    objectives[:, -1] = np.sin(theta)
    return objectives
