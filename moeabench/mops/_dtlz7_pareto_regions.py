# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Analytical Pareto-record parameter domain for DTLZ7-like benchmarks.

For g=1, the last DTLZ7-like base objective is

    f_last = 2*M - sum_i a(x_i),  a(x) = x*(1+sin(3*pi*x)).

Since each of the first M-1 objectives equals x_i, an x_i is globally
Pareto-optimal iff a(x_i) is a *strict prefix record* of a over [0,x_i].
Otherwise replacing x_i by a smaller t with a(t)>=a(x_i) dominates it.

The record set on [0,1] is exactly the two disconnected intervals:
    [0, first_local_max] U (record_reentry, second_local_max].

This proof also applies to DPF2's D-objective base, which is followed
only by nonnegative coordinatewise monotone redundant objectives.
Sampling is uniform in the *allowed decision parameters*, NOT in
objective-space arc length/surface area. No baseline or GT is changed.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq


def _benefit(x):
    return x * (1.0 + np.sin(3.0 * np.pi * x))


def _benefit_derivative(x):
    return 1.0 + np.sin(3.0 * np.pi * x) + 3.0 * np.pi * x * np.cos(3.0 * np.pi * x)


@lru_cache(maxsize=1)
def dtlz7_record_intervals():
    """Return exact numerical boundaries of the two Pareto-record intervals.

    A derivative root fixes each record maximum. The reentry point is
    where the second increasing segment first exceeds the first peak's
    record value. The second interval is open at its lower boundary.
    """
    peak1 = brentq(_benefit_derivative, 0.20, 0.30, xtol=1e-14)
    peak2 = brentq(_benefit_derivative, 0.80, 0.90, xtol=1e-14)
    reentry = brentq(lambda x: _benefit(x) - _benefit(peak1),
                     0.55, 0.75, xtol=1e-14)
    if not (0.0 < peak1 < 0.50 < reentry < peak2 < 1.0):
        raise RuntimeError("Invalid analytical DTLZ7 Pareto-record intervals")
    return ((0.0, float(peak1)), (float(reentry), float(peak2)))


def map_uniform_to_record_intervals(uniform):
    """Map uniform [0,1] values to the allowed Pareto parameter intervals.

    Preserves equal density per unit *decision-parameter* length. For the
    unique interval join, return the first maximum, never the dominated
    second-interval entry point. Numerical values arbitrarily close to the
    join require a strict-record margin to avoid floating-point ties.
    """
    u = np.asarray(uniform, dtype=np.float64)
    if np.any(~np.isfinite(u)) or np.any((u < 0.0) | (u > 1.0)):
        raise ValueError("Expected finite uniform variates in [0,1]")
    (lo1, hi1), (lo2, hi2) = dtlz7_record_intervals()
    span1 = hi1 - lo1
    span2 = hi2 - lo2
    length = span1 + span2
    cumulative = u * length
    result = np.where(cumulative <= span1,
                      lo1 + cumulative,
                      lo2 + (cumulative - span1))
    # When float arithmetic makes an essentially equal record after the
    # reentry point, nudge onto the strictly improving branch.
    upper = cumulative > span1
    result = np.where(upper, np.maximum(result, lo2 + 1e-9), result)
    return result


def sample_dtlz7_record_parameters(n_points, n_dimensions, *,
                                   seed=42, structured=False):
    """Deterministic Pareto-set decision parameters.

    Structured, one-dimensional mode extends the existing DPF2 grid into
    the true disconnected domain. Otherwise use the existing private
    RandomState(42) policy, conditioned on membership in the record set.
    """
    if (not isinstance(n_points, (int, np.integer)) or n_points < 1 or
            not isinstance(n_dimensions, (int, np.integer)) or n_dimensions < 1):
        raise ValueError("Need n_points>=1 and n_dimensions>=1")
    if structured:
        if n_dimensions != 1:
            raise ValueError("Structured grid requires one decision parameter")
        if n_points == 1:
            samples = np.array([[0.5]])
        else:
            samples = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)
    else:
        samples = np.random.RandomState(seed).random((n_points,n_dimensions))
    return map_uniform_to_record_intervals(samples)
