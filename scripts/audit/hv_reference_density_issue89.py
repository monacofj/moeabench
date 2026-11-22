# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent 2D finite-reference HV density experiment (#89).

This is research evidence, NOT a new public HV definition or a replacement
calibration policy. It reads the actual MOP PF samplers but does not write
release baselines or modify the existing tests and certification records.

Run: python -m scripts.audit.hv_reference_density_issue89
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import moeabench as mb


def independent_hv2(front, ideal, nadir):
    """Exact 2D dominated area in normalized coordinates (no HV backend).

    A sweep ordered by ascending x accumulates non-overlapping vertical
    rectangles, retaining only improvements of the smallest y encountered.
    It is valid for finite minimization fronts inside the reference box.
    """
    f = np.asarray(front, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)
    if f.ndim != 2 or f.shape[1] != 2 or len(f) == 0:
        raise ValueError("A nonempty 2D objective front is required.")
    span = nadir - ideal
    if np.any(span <= 0) or not np.all(np.isfinite(f)):
        raise ValueError("Nondegenerate finite normalization bounds are required.")

    p = (f - ideal) / span
    p = p[np.argsort(p[:, 0], kind="stable")]
    ref = 1.1
    best_y = ref
    area = 0.0
    for x, y in p:
        if x < ref and y < best_y:
            area += (ref - x) * (best_y - y)
            best_y = y
    return float(area)


def insert_true_front_midpoint(name, gt):
    """Add one analytically Pareto-optimal point in the largest x gap."""
    ordered = np.asarray(gt, dtype=float)
    ordered = ordered[np.argsort(ordered[:, 0])]
    k = int(np.argmax(np.diff(ordered[:, 0])))
    a, b = ordered[k], ordered[k + 1]
    if name == "DTLZ1":
        x = (a[0] + b[0]) / 2
        midpoint = np.array([x, 0.5 - x])
    elif name == "DTLZ2":
        theta = (np.arctan2(a[1], a[0])
                 + np.arctan2(b[1], b[0])) / 2
        midpoint = np.array([np.cos(theta), np.sin(theta)])
    else:
        raise ValueError("Only two objectively proved continuous fronts are used.")
    return midpoint


def evaluate_case(name, n, *, check_public=True):
    """Return independent-area and public-API comparisons for one budget."""
    mop = getattr(mb.mops, name)(M=2)
    gt = np.asarray(mop.pf(n_points=n), dtype=float)
    if gt.shape != (n, 2):
        raise AssertionError("PF sampler returned the wrong shape")
    if name == "DTLZ1":
        np.testing.assert_allclose(gt.sum(axis=1), .5, atol=1e-10, rtol=0)
    else:
        np.testing.assert_allclose(np.sum(gt**2, axis=1), 1.0,
                                   atol=1e-10, rtol=0)
    ideal, nadir = gt.min(axis=0), gt.max(axis=0)
    if not np.allclose(ideal, [0, 0], atol=1e-12) or not np.allclose(
            nadir, [.5, .5] if name == "DTLZ1" else [1, 1],
            atol=1e-12):
        raise AssertionError("Expected analytical endpoints are missing")
    midpoint = insert_true_front_midpoint(name, gt)
    evaluated = np.vstack((gt, midpoint))

    denominator = independent_hv2(gt, ideal, nadir)
    numerator = independent_hv2(evaluated, ideal, nadir)
    expected = numerator / denominator
    if not numerator > denominator:
        raise AssertionError("The inserted Pareto point must add some HV")

    record = {
        "problem": name, "gt_points": n,
        "independent_hv_gt": denominator,
        "independent_hv_evaluated": numerator,
        "independent_h_abs": expected,
        "increment": numerator - denominator,
        "point_on_true_pf": midpoint.tolist(),
    }
    if check_public:
        class OwnedFront(np.ndarray):
            pass

        with TemporaryDirectory(prefix="moeabench-hv89-") as work:
            mop.calibrate(source_baseline=Path(work) / "calibration.json",
                          source_gt=gt)
            front = evaluated.view(OwnedFront)
            front.mop = mop
            actual = mb.metrics.hv(
                front, ref=gt, scale="abs", mode="exact", progress=False,
            )
            raw = mb.metrics.hv(
                evaluated, ref=gt, scale="raw", mode="exact", progress=False,
            )
            baseline = mb.metrics.hv(
                gt, ref=gt, scale="raw", mode="exact", progress=False,
            )
        np.testing.assert_allclose(float(raw), numerator, rtol=2e-10, atol=1e-12)
        np.testing.assert_allclose(float(baseline), denominator,
                                   rtol=2e-10, atol=1e-12)
        np.testing.assert_allclose(float(actual), expected,
                                   rtol=2e-10, atol=1e-12)
        record["public_h_abs"] = float(actual)
    return record


def main():
    cases = [("DTLZ1", 2000), ("DTLZ1", 10000),
             ("DTLZ2", 2000)]
    print(json.dumps({
        "study": "issue-89-independent-2d-exact-hv",
        "scope": "true-PF insertion into exact MOP sampler",
        "note": "not a Monte Carlo study or a performance certification",
        "cases": [evaluate_case(name, n) for name, n in cases],
    }, indent=2))


if __name__ == "__main__":
    main()
