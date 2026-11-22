# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact independent DPF1(M=3,D=2) degenerate-front HV study (#89).

This is a research oracle for a *specific proved geometry*, not an
implementation of a general 3D HV metric or a GT calibration replacement.

Run: python -m scripts.audit.hv_dpf1_degenerate_issue89
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import moeabench as mb


def independent_dpf1_hv3(front, ideal, nadir):
    """Swept 3D dominated volume for normalized DPF1=(t,1-t,1-t).

    Across x slabs the latest point dominates previous active points in
    both y and z. Its yz rectangle therefore defines the *entire*
    cross-section. This is not a general 3D sweep oracle.
    """
    f = np.asarray(front, dtype=float)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("Expected a nonempty three-objective front")
    ideal, nadir = np.asarray(ideal, dtype=float), np.asarray(nadir, dtype=float)
    span = nadir - ideal
    if np.any(span <= 0):
        raise ValueError("Reference must contain all analytical endpoints")
    p = (f - ideal) / span
    p = p[np.argsort(p[:, 0], kind="stable")]
    if not np.allclose(p[:, 1], 1 - p[:, 0], rtol=0, atol=1e-10):
        raise ValueError("The DPF1 normalized curve assumption does not hold")
    if not np.allclose(p[:, 2], p[:, 1], rtol=0, atol=1e-10):
        raise ValueError("The projected third objective is not redundant as assumed")
    if np.any(np.diff(p[:, 0]) < 0) or np.any(np.diff(p[:, 1]) > 1e-10):
        raise ValueError("Nonmonotone points invalidate the 3D slice proof")

    xs = p[:, 0]
    next_x = np.r_[xs[1:], 1.1]
    areas = (1.1 - p[:, 1]) * (1.1 - p[:, 2])
    return float(np.sum((next_x - xs) * areas))


def exact_true_dpf1_hv3():
    """Integral of (x+0.1)^2 for x=0..1, plus last .1 full slab."""
    return float((1.1**3 - 0.1**3) / 3 + 0.1 * 1.1**2)


def evaluate_case(n=10000, *, check_public=False):
    """Use the real DPF1 sampler and compare fixed-box finite/true HV."""
    mop = mb.mops.DPF1(M=3, D=2)
    gt = np.asarray(mop.pf(n_points=n), dtype=float)
    if gt.shape != (n, 3):
        raise AssertionError("DPF1 sampler returned an unexpected objective shape")
    ideal, nadir = gt.min(axis=0), gt.max(axis=0)
    order = np.argsort(gt[:, 0])
    gt_sorted = gt[order]
    # The final interval has the largest positive projected yz area.
    p = (gt_sorted[-2] + gt_sorted[-1]) / 2
    evaluated = np.vstack((gt, p))

    denominator = independent_dpf1_hv3(gt, ideal, nadir)
    numerator = independent_dpf1_hv3(evaluated, ideal, nadir)
    true_hv = exact_true_dpf1_hv3()
    if not denominator < numerator < true_hv:
        raise AssertionError("Finite sample/true front HV ordering failed")
    result = {
        "problem": "DPF1", "M": 3, "D": 2, "gt_points": n,
        "independent_hv_gt": denominator,
        "independent_hv_evaluated": numerator,
        "sampled_gt_h_abs": numerator / denominator,
        "true_continuous_hv": true_hv,
        "continuous_bound_ratio": numerator / true_hv,
        "increment": numerator - denominator,
    }
    if check_public:
        class OwnedFront(np.ndarray):
            pass

        with TemporaryDirectory(prefix="moeabench-dpf1hv89-") as work:
            mop.calibrate(source_baseline=Path(work) / "calibration.json",
                          source_gt=gt)
            front = evaluated.view(OwnedFront)
            front.mop = mop
            absolute = mb.metrics.hv(
                front, ref=gt, scale="abs", mode="exact", progress=False
            )
            raw = mb.metrics.hv(
                evaluated, ref=gt, scale="raw", mode="exact", progress=False
            )
            baseline = mb.metrics.hv(
                gt, ref=gt, scale="raw", mode="exact", progress=False
            )
        np.testing.assert_allclose(float(raw), numerator,
                                   rtol=2e-9, atol=1e-11)
        np.testing.assert_allclose(float(baseline), denominator,
                                   rtol=2e-9, atol=1e-11)
        np.testing.assert_allclose(float(absolute), numerator / denominator,
                                   rtol=2e-9, atol=1e-11)
        result["public_h_abs"] = float(absolute)
    return result


def main():
    print(json.dumps({
        "study": "issue-89-independent-degenerate-dpf1-3d-hv",
        "note": "not an API change or numerical/performance certification",
        "cases": [evaluate_case(2000, check_public=True),
                  evaluate_case(10000, check_public=False)],
    }, indent=2))


if __name__ == "__main__":
    main()
