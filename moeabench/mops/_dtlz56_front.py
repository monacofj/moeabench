# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Complete-front construction for the original DTLZ5/DTLZ6 formulations.

A globally bounded weighted Chebyshev method explores all positive-efficiency
regions, including nonconvex g>0 components, without replacing original
evaluation equations. At fixed g, angular-patch box feasibility and scale
have closed-form O(M) evaluations. Branch-and-bound over the scalar g provides
an explicit absolute scalarization tolerance. Finite objective-space sampling
provides an approximation; it cannot prove continuous finite-sample coverage.
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
    # Scalarization weights are homogeneous; normalize to a canonical
    # scale, but never silently change the user's preference direction.
    # Extreme ratios cannot be certified to an absolute gap in finite
    # precision. The finite GT sampler explicitly conditions its rays.
    raw = _check_weights(weights)
    w = raw / float(np.max(raw))
    if np.min(w) < 1e-7:
        raise ValueError(
            "Ill-conditioned positive Chebyshev weights: minimum relative "
            "weight must be >= 1e-7; explicitly condition the weights first"
        )
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

    # Reconstruct the optimum directly from the exact angular-capacity
    # recurrence. Avoid adding an absolute epsilon to a potentially large
    # Chebyshev scale: that fails for highly nonuniform positive weights.
    a = pi / (4.0 * (1.0 + best_g))
    b = pi * (1.0 + 2.0 * best_g) / (4.0 * (1.0 + best_g))
    capacity = float(w[0])
    inner_angles = []
    for j in range(1, len(w) - 1):
        cap = float(w[j])
        raw_angle = atan2(cap, capacity)
        angle = max(a, min(b, raw_angle))
        if raw_angle < a:
            capacity = cap / sin(a)
        elif raw_angle > b:
            capacity = capacity / cos(b)
        else:
            capacity = hypot(capacity, cap)
        inner_angles.append(angle)

    angles = np.r_[atan2(w[-1], capacity), inner_angles[::-1]]
    cosines, sines = np.cos(angles), np.sin(angles)
    objectives = np.zeros(len(w))
    for i in range(len(w)):
        val = 1.0 + best_g
        if i < len(w) - 1:
            val *= np.prod(cosines[:len(w) - i - 1])
        if i > 0:
            val *= sines[len(w) - i - 1]
        objectives[i] = val

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


def sample_original_dtlz56_ps(mop, n_points, *, scalar_tol=1e-7):
    """Deterministically sample the original complete-front components.

    n_points is exact. The positive-weight scalarization parameterizes every
    strictly positive Pareto objective vector; explicit g=0 anchors preserve
    the degenerate part and both endpoints. The objective-space maximin
    selection spreads a finite sample across the resulting mixed-dimensional
    geometry. Returns *decision vectors* (original evaluation is unchanged).

    Every scalarization is globally bounded to scalar_tol. This does not
    imply exact uniformity or a zero-distance Hausdorff certificate.
    """
    from scipy.stats import qmc

    if not isinstance(n_points, (int, np.integer)) or n_points < 1:
        raise ValueError("n_points must be a positive integer")
    m, k = mop.M, mop.K
    if m < 4 or k < 1:
        raise ValueError("Complete-front sampler is intended for M >= 4")
    name = mop.__class__.__name__
    if name not in ("DTLZ5", "DTLZ6"):
        raise ValueError("Sampler requires original DTLZ5 or DTLZ6")
    g_max = (k / 4.0) if name == "DTLZ5" else float(k)

    n_curve = n_points if n_points <= 2 else min(
        max(2, int(2.0 * np.sqrt(n_points))),
        max(2, n_points // 4),
    )
    anchors = np.zeros((n_curve, mop.N), dtype=float)
    anchors[:, 0] = np.linspace(0.0, 1.0, n_curve)
    anchors[:, 1:m-1] = 0.5
    anchors[:, m-1:] = 0.5 if name == "DTLZ5" else 0.0

    if n_points == n_curve:
        return anchors

    # Deterministic Sobol directions from exactly realizable minimal-g
    # boundary faces. At least one angular decision variable is at 0 or 1.
    # Independent positive Gaussian rays explore complementary preference
    # directions and prevent a sampler tied to just one boundary measure.
    pool_size = max(4 * n_points, 256)
    sobol = qmc.Sobol(d=m, scramble=True, seed=42 + m)
    sobol_points = sobol.random_base2(
        int(np.ceil(np.log2(pool_size)))
    )[:pool_size]
    x = np.zeros((pool_size, mop.N), dtype=float)
    x[:, 0] = sobol_points[:, 0]
    x[:, 1:m-1] = sobol_points[:, 2:m]
    face_indices = np.arange(pool_size) % (m - 2)
    x[np.arange(pool_size), 1 + face_indices] = (
        np.arange(pool_size) % 2
    )
    # Include the *full* original feasible g range; otherwise extreme
    # nondegenerate Pareto components could be systematically omitted.
    g_limit = np.where(
        (np.arange(pool_size) % 4) == 0,
        g_max,
        np.minimum(g_max, 1.5),
    )
    g = g_limit * sobol_points[:, 1]**2
    x[:, m-1:] = (
        (0.5 + np.sqrt(g / k))[:, None]
        if name == "DTLZ5"
        else ((g / k)**10)[:, None]
    )
    boundary_rays = np.maximum(mop.evaluation(x)["F"], 1e-12)

    rng = np.random.default_rng(42 + m)
    gaussian_rays = np.abs(rng.standard_normal((pool_size // 2, m))) + 1e-10
    sparse_rays = rng.gamma(shape=0.35, scale=1.0, size=(pool_size // 4, m))
    weights = np.vstack((boundary_rays, gaussian_rays, sparse_rays))
    # The finite sample approaches zero-valued boundary rays through a
    # stated positive weight floor, rather than changing the scalarization
    # weights invisibly. Exact g=0 endpoints are separately anchored.
    weights = np.maximum(
        weights, 1e-7 * np.max(weights, axis=1, keepdims=True)
    )
    candidates = [
        scalarized_candidate(w, g_max, abs_tol=scalar_tol)
        for w in weights
    ]
    generated_ps = np.asarray([
        decision_vector(c, M=m, K=k, problem=name)
        for c in candidates
    ])
    generated_pf = np.asarray([c.objectives for c in candidates])

    # Spread points in normalized objective space. The anchors are included
    # unconditionally; all other points are selected by deterministic
    # greedy farthest-point sampling, without applying a GT-dependent
    # indicator as the selection objective.
    known_pf = mop.evaluation(anchors)["F"]
    full_pf = np.vstack((known_pf, generated_pf))
    full_ps = np.vstack((anchors, generated_ps))
    scales = np.maximum(np.ptp(full_pf, axis=0), 1e-12)

    chosen = list(range(n_curve))
    remaining = full_pf[n_curve:]
    min_dist = np.min(
        np.sum(
            ((remaining[:, None, :] - known_pf[None, :, :]) / scales)**2,
            axis=2,
        ),
        axis=1,
    )
    for _ in range(n_points - n_curve):
        idx = int(np.argmax(min_dist))
        chosen.append(n_curve + idx)
        distance = np.sum(((remaining - remaining[idx]) / scales)**2, axis=1)
        min_dist = np.minimum(min_dist, distance)
        min_dist[idx] = -1.0

    return full_ps[chosen]
