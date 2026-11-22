# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent finite-reference Hypervolume counterexamples for issue #89.

These tests document the EXISTING, unclipped HV contracts. They do not
authorize metric, ADR, regression-fixture, or numerical-baseline changes.
"""

from __future__ import annotations

import numpy as np
import pytest

import moeabench as mb


# Both sets lie on the exact DTLZ1(M=2), g=0 Pareto line f1+f2=1/2.
FINITE_GT = np.array([[0.0, 0.5], [0.5, 0.0]])
MIDPOINT = np.array([[0.25, 0.25]])


def _raw(front):
    return float(mb.metrics.hv(
        front, ref=FINITE_GT, mode="exact", scale="raw", progress=False
    ))


def test_absolute_hv_can_exceed_one_without_exceeding_true_pareto_front(tmp_path):
    """A sparse GT has less dominated area than another true-PF sample."""
    assert np.allclose(FINITE_GT.sum(axis=1), 0.5)
    assert np.allclose(MIDPOINT.sum(axis=1), 0.5)

    mop = mb.mops.DTLZ1(M=2)
    mop.calibrate(source_gt=FINITE_GT, source_baseline=tmp_path / "gt.json")

    class MopOwnedFront(np.ndarray):
        pass

    evaluated = MIDPOINT.copy().view(MopOwnedFront)
    evaluated.mop = mop

    # The exact normalized HV reference point is (1.1, 1.1).
    # Endpoints cover .11 + .11 - .01 = .21; midpoint covers .6*.6=.36.
    assert _raw(FINITE_GT) == pytest.approx(0.21)
    assert _raw(MIDPOINT) == pytest.approx(0.36)

    result = mb.metrics.hv(
        evaluated, ref=FINITE_GT, mode="exact", scale="abs", progress=False
    )
    assert float(result) == pytest.approx(12.0 / 7.0)
    assert result.diagnostics["hv_abs_gt_points"] == 2
    assert result.diagnostics["hv_abs_denominator"] == pytest.approx(0.21)


def test_relative_hv_can_exceed_one_against_a_weaker_external_reference():
    """ADR 0047 permits contextual competitive ratios above one."""
    result = mb.metrics.hv(
        MIDPOINT, ref=FINITE_GT, mode="exact", scale="rel", progress=False
    )
    assert _raw(MIDPOINT) > _raw(FINITE_GT)
    assert float(result) == pytest.approx(12.0 / 7.0)
