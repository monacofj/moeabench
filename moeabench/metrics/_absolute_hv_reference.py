# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Proved continuous-front denominators for scalar absolute HV.

Only the exact built-in original MOP classes and the precisely proved
axis-normalized geometry are eligible. This module does not calibrate GT
or select a different reference geometry.
"""

from math import factorial, gamma, pi

import numpy as np

from ..mops.DTLZ1 import DTLZ1
from ..mops.DTLZ2 import DTLZ2
from ..mops.DTLZ3 import DTLZ3
from ..mops.DTLZ4 import DTLZ4
from ..mops.DPF1 import DPF1


def exact_continuous_hv(mop, ideal, nadir):
    """Return a proved denominator or None when applicability is unverified.

    Calibration provenance must explicitly indicate analytical sampling.
    External/empirical or unspecified GT always uses sampled-GT fallback.
    An exact built-in class and axis extrema are checked independently.
    """
    if getattr(mop, "_gt_provenance", None) != "analytical":
        return None

    kind = type(mop)
    m = mop.M
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)
    if (not isinstance(m, (int, np.integer)) or m < 2
            or ideal.shape != (m,) or nadir.shape != (m,)
            or not np.all(np.isfinite(ideal))
            or not np.all(np.isfinite(nadir))):
        return None

    if kind is DTLZ1:
        target_ideal = np.zeros(m)
        target_nadir = np.full(m, .5)
        complement = 1 / factorial(m)
    elif kind in (DTLZ2, DTLZ3, DTLZ4):
        target_ideal = np.zeros(m)
        target_nadir = np.ones(m)
        complement = pi**(m / 2) / (2**m * gamma(m / 2 + 1))
    elif kind is DPF1 and m == 3 and mop.D == 2:
        # Base simplex: (0.5*t, 0.5*(1-t)).
        # Redundant f3 = w0*f1+w1*f2; w0 < w1 yields normalized
        # (t, 1-t, 1-t) after per-objective true extremal scaling.
        weights = np.asarray(mop._chaos_pool[:2], dtype=float)
        if (weights.shape != (2,)
                or not np.all(np.isfinite(weights))
                or not 0 < weights[0] < weights[1]):
            return None
        target_ideal = np.array([0., 0., .5 * weights[0]])
        target_nadir = np.array([.5, .5, .5 * weights[1]])
        if not (np.allclose(ideal, target_ideal, atol=1e-12, rtol=1e-12)
                and np.allclose(nadir, target_nadir,
                                atol=1e-12, rtol=1e-12)):
            return None
        return float(((1.1**3 - .1**3) / 3) + .1 * 1.1**2)
    else:
        return None

    if (not np.allclose(ideal, target_ideal, atol=1e-12, rtol=1e-12)
            or not np.allclose(nadir, target_nadir,
                               atol=1e-12, rtol=1e-12)):
        return None

    exact = 1.1**m - complement
    if not np.isfinite(exact) or exact <= 0:
        return None
    return float(exact)
