# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic, globally bounded Chebyshev research solver for DTLZ5/DTLZ6.

This is NOT the canonical analytical PS/PF generator. It solves a *single*
positive-weight scalarization to a requested absolute objective-scale bound.
Every strictly positive original Pareto objective vector is supported by some
positive Chebyshev weight vector, including nonconvex front regions. Finite
weights and finite accuracy do not by themselves certify global front coverage.

At a fixed g, the entire angular patch has an exact O(M) box-feasibility
oracle. For g in [a,b], a patch with radius 1+a and angular envelope b is a
componentwise *optimistic relaxation* of all original patches in the interval.
A binary search on the relaxed box gives a valid global scalarization lower
bound, enabling deterministic interval branch-and-bound over one variable g.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import atan2, cos, hypot, pi, sin, sqrt

import numpy as np


@dataclass(frozen=True)
class ScalarizedCandidate:
    g: float
    scale: float
    objectives: np.ndarray
    lower_bound: float
    abs_gap: float
    nodes: int


def _check_weights(weights):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) < 4 or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Expect >=4 strictly positive finite objective weights")
    return w


def _relaxed_feasible(caps, radius, angular_g, *, return_objectives=False):
    """Angular-patch intersection with objective upper box, exact up to FP."""
    caps = np.asarray(caps, dtype=float)
    if caps[-1] >= radius:
        if return_objectives:
            v = np.zeros_like(caps)
            v[-1] = radius
            return v
        return True

    if np.any(caps < 0.0):
        return None if return_objectives else False

    f = np.empty_like(caps)
    f[-1] = caps[-1]
    rem = sqrt(max(0.0, radius * radius - f[-1] * f[-1]))
    lower = pi / (4.0 * (1.0 + angular_g))
    upper = pi * (1.0 + 2.0 * angular_g) / (4.0 * (1.0 + angular_g))
    sine_lower, sine_upper = sin(lower), sin(upper)

    for j in range(len(caps) - 2, 0, -1):
        if rem * sine_lower > caps[j] + 1e-14:
            return None if return_objectives else False
        f[j] = min(caps[j], rem * sine_upper)
        rem = sqrt(max(0.0, rem * rem - f[j] * f[j]))

    if rem > caps[0] + 1e-14:
        return None if return_objectives else False
    f[0] = rem
    return f if return_objectives else True


def _bounded_scale(w, g_radius, g_angle):
    """Closed-form O(M) scale for the entire angular patch.

    For the first two objectives, the largest reachable 2D radius under
    caps (w0,w1) is obtained by clamping atan2(w1,w0) into [a,b]. Recursively
    combine that radius with each further objective cap w_j using the same
    pair operation. The final theta_1 is unrestricted, hence the last step
    is hypot(capacity, w[-1]). This gives t=(1+g_radius)/capacity exactly,
    up to FP rounding, without binary search over t.

    The small symmetric enclosure allows the interval B&B to retain its
    conservative global lower-bound behavior in floating-point arithmetic.
    """
    a = pi / (4.0 * (1.0 + g_angle))
    b = pi * (1.0 + 2.0 * g_angle) / (4.0 * (1.0 + g_angle))
    capacity = float(w[0])
    for j in range(1, len(w) - 1):
        cap = float(w[j])
        ideal_angle = atan2(cap, capacity)
        if ideal_angle < a:
            capacity = cap / sin(a)
        elif ideal_angle > b:
            capacity = capacity / cos(b)
        else:
            capacity = hypot(capacity, cap)
    exact_scale = (1.0 + g_radius) / hypot(capacity, float(w[-1]))
    return max(0.0, exact_scale - 1e-12), exact_scale + 1e-12


def scalarized_candidate(
    weights, max_g, *, abs_tol=1e-5, max_nodes=12000
) -> ScalarizedCandidate:
    """Solve a positive Chebyshev scalarization with certified scalar gap.

    The returned gap bounds the difference between achieved max_i(f_i/w_i)
    and the global minimum over the entire original feasible DTLZ5/6 image
    (provided g in [0,max_g]). This does NOT by itself certify that every
    returned point is exactly globally Pareto optimal or that a finite
    collection covers the whole front.

    Original benchmark limits: DTLZ5 max_g=K/4, DTLZ6 max_g=K.
    """
    w = _check_weights(weights)
    if not np.isfinite(max_g) or max_g <= 0:
        raise ValueError("max_g must be positive and finite")
    if not np.isfinite(abs_tol) or abs_tol <= 0:
        raise ValueError("abs_tol must be positive and finite")
    if max_nodes < 1:
        raise ValueError("max_nodes must be >=1")

    def fixed(g):
        return _bounded_scale(w, g, g)[1]

    best_g = 0.0
    best = fixed(0.0)
    # A feasible vector with radius (1+g) must have scale >= (1+g)/||w||.
    limit = min(max_g, max(0.0, best * np.linalg.norm(w) - 1.0))
    for g in np.linspace(0.0, limit, 9):
        val = fixed(float(g))
        if val < best:
            best, best_g = val, float(g)

    def interval_lower(a, b):
        # Radius 1+a is no larger than any radius in [a,b]; angles allowed
        # at b contain the angular ranges at every g in [a,b].
        return _bounded_scale(w, a, b)[0] - 2e-13

    heap = [(interval_lower(0.0, limit), 0.0, limit)]
    nodes = 0

    while heap:
        bound, a, b = heappop(heap)
        if bound >= best - abs_tol:
            continue
        if nodes >= max_nodes:
            raise RuntimeError(
                "Global scalarization bound not certified within max_nodes"
            )
        middle = (a + b) / 2.0
        nodes += 1
        value = fixed(middle)
        if value < best:
            best, best_g = value, middle

        for left, right in ((a, middle), (middle, b)):
            lb = interval_lower(left, right)
            if lb < best - abs_tol:
                heappush(heap, (lb, left, right))

    radius = 1.0 + best_g
    # Use a negligible upward slack solely to reconstruct angles reliably;
    # the returned achieved scalarization is always recomputed from f.
    objectives = _relaxed_feasible(
        (best + 2e-12) * w, radius, best_g, return_objectives=True
    )
    if objectives is None:
        raise ArithmeticError("Could not reconstruct feasible scalarized vector")
    actual = float(np.max(objectives / w))
    return ScalarizedCandidate(
        g=best_g,
        scale=actual,
        objectives=objectives,
        lower_bound=max(0.0, best - abs_tol - 4e-12),
        abs_gap=abs_tol + 4e-12 + max(0.0, actual - best),
        nodes=nodes,
    )


def decision_vector(candidate, *, M, K, problem):
    """Recover a valid original decision vector from a scalarized candidate.

    This is a research helper, not a public reference-front producer.
    """
    if M < 4 or K < 1 or problem not in ("DTLZ5", "DTLZ6"):
        raise ValueError("Require original DTLZ5/6 with M>=4, K>=1")
    f = np.asarray(candidate.objectives)
    if f.shape != (M,):
        raise ValueError("candidate dimension does not match M")
    g = float(candidate.g)
    if g < 0 or g > (K / 4.0 if problem == "DTLZ5" else float(K)) + 1e-12:
        raise ValueError("Candidate g outside the original problem range")

    x = np.zeros(M + K - 1)
    for i in range(M - 1):
        # Inverse of the original positive hypersphere parameterization.
        index = M - i - 1
        theta = np.arctan2(f[index], np.linalg.norm(f[:index]))
        if i == 0:
            x[i] = theta / (pi / 2.0)
        elif g == 0.0:
            x[i] = 0.5
        else:
            x[i] = ((4.0 * (1.0 + g) / pi) * theta - 1.0) / (2.0 * g)
    if problem == "DTLZ5":
        x[M - 1:] = 0.5 + sqrt(g / K)
    else:
        x[M - 1:] = (g / K) ** 10
    # Correction solely for floating point near a domain boundary.
    if np.any(x < -5e-9) or np.any(x > 1.0 + 5e-9):
        raise ArithmeticError("Reconstructed decision vector is infeasible")
    return np.clip(x, 0.0, 1.0)
