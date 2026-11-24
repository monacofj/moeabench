# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New tests for independent low-dimensional DTLZ5/6 zero-gap theorem.

These tests do not modify any pre-existing regression or frozen baseline.
"""
from math import cos, pi, sin, sqrt

import numpy as np
import pytest

from moeabench.mops._dtlz56_lowdim_curve import (
    audit_low_dimensional_g0_dominator,
)


def _original_objectives(m, g, theta1, theta2=None):
    """Reconstruct the published M=2/3 equations without a production MOP."""
    radius = 1.0 + g
    if m == 2:
        return np.array([radius*cos(theta1), radius*sin(theta1)])
    return np.array([
        radius*cos(theta1)*cos(theta2),
        radius*cos(theta1)*sin(theta2),
        radius*sin(theta1),
    ])


@pytest.mark.parametrize("problem,k", [("DTLZ5", 4), ("DTLZ6", 4)])
@pytest.mark.parametrize("m", [2, 3])
@pytest.mark.parametrize("theta1", [0., pi/4, pi/2])
@pytest.mark.parametrize("fraction", [0.01, 0.25, 0.9])
def test_feasible_positive_g_is_dominated_without_g_gap(
    problem, k, m, theta1, fraction
):
    maximum = k/4 if problem == "DTLZ5" else k
    g = maximum*fraction
    low = pi/(4*(1+g))
    high = pi/2-low
    angles = [None] if m == 2 else [low, (low+high)/2, high]
    for theta2 in angles:
        y = _original_objectives(m, g, theta1, theta2)
        result = audit_low_dimensional_g0_dominator(
            y, problem=problem, k=k, tolerance=1e-10
        )
        assert result.status == "witness"
        z = np.asarray(result.witness)
        assert z.shape == y.shape
        assert np.all(z <= y+1e-10)
        assert np.any(z < y-1e-10)
        assert np.isclose(np.linalg.norm(z), 1.0, atol=1e-12)
        assert np.linalg.norm(z) < np.linalg.norm(y)-1e-10


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [2, 3])
def test_g_zero_is_not_misreported_as_strictly_dominated(problem, m):
    y = _original_objectives(m, 0., pi/3, pi/4)
    result = audit_low_dimensional_g0_dominator(y, problem=problem, k=3)
    assert result.status == "unverified"
    assert result.witness is None


@pytest.mark.parametrize("m", [2, 3])
def test_invalid_image_rejected(m):
    y = np.ones(m)*3
    with pytest.raises(ValueError, match="outside original feasible image"):
        audit_low_dimensional_g0_dominator(y, problem="DTLZ5", k=2)


def test_original_evaluator_not_reused(monkeypatch):
    from moeabench.mops.DTLZ5 import DTLZ5
    from moeabench.mops.DTLZ6 import DTLZ6

    def forbidden(*_args, **_kwargs):
        raise AssertionError("production evaluator/sampler called")

    for klass in (DTLZ5, DTLZ6):
        monkeypatch.setattr(klass, "evaluation", forbidden)
        monkeypatch.setattr(klass, "ps", forbidden)
        monkeypatch.setattr(klass, "pf", forbidden)
    y = _original_objectives(3, 0.2, pi/4, pi/4)
    assert audit_low_dimensional_g0_dominator(
        y, problem="DTLZ5", k=4
    ).status == "witness"


@pytest.mark.parametrize("bad", [
    [float("nan"), 1.], [1., -0.1], [1., 0., 0., 0.],
])
def test_malformed_objectives_rejected(bad):
    with pytest.raises(ValueError):
        audit_low_dimensional_g0_dominator(bad, problem="DTLZ6", k=3)
