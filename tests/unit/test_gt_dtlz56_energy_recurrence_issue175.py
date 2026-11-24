# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent tests for squared-energy DTLZ5/6 fixed-g feasibility."""
from math import cos, pi, sin, sqrt

import numpy as np
import pytest

from scripts.audit.dtlz56_energy_recurrence import (
    observe_fixed_g_energy_feasibility,
)


def _independent_angle_feasibility(y, g):
    """Separate geometrical allocation in lengths, not squared recurrence."""
    radius = 1.0 + g
    if y[-1] >= radius:
        return True
    residual = sqrt(radius**2-y[-1]**2)
    low = pi/(4*radius)
    high = pi/2-low
    for j in range(len(y)-2, 0, -1):
        if residual*sin(low) > y[j]:
            return False
        contribution = min(y[j], residual*sin(high))
        residual = sqrt(max(0.0, residual**2-contribution**2))
    return residual <= y[0]


def _original_feasible_point(m, g, first_angle, inner_fraction):
    radius = 1+g
    lower = pi/(4*radius)
    upper = pi/2-lower
    angles = [first_angle]+[lower+(upper-lower)*inner_fraction]*(m-2)
    out = np.ones(m)*radius
    for j in range(m):
        for i in range(m-j-1):
            out[j] *= cos(angles[i])
        if j:
            out[j] *= sin(angles[m-j-1])
    return out


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("g", [0., 0.125, 0.5])
@pytest.mark.parametrize("first", [0., pi/4, pi/2])
def test_exact_original_feasible_objectives_fit(problem, m, g, first):
    y = _original_feasible_point(m, g, first, 0.5)
    outcome = observe_fixed_g_energy_feasibility(
        y+1e-10, g=g, problem=problem, k=4)
    assert outcome.observation == "numerically_feasible"
    assert outcome.to_dict()["scientific_status"] == "unverified"


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [2, 3, 4, 5, 10])
def test_independent_length_predicate_matches_squared_predicate(problem, m):
    rng = np.random.default_rng(5212+m)
    gmax = 0.5
    # Continuous test observations, not scientific oracle certification.
    for _ in range(64):
        g = float(rng.uniform(0.01, gmax))
        y = rng.uniform(0.0, 2.0, m)
        actual = observe_fixed_g_energy_feasibility(
            y, g=g, problem=problem, k=4)
        expected = _independent_angle_feasibility(y, g)
        assert (actual.observation == "numerically_feasible") == expected


def test_excludes_known_insufficient_caps():
    for m in (2, 3, 4, 5, 10):
        y = np.zeros(m)
        outcome = observe_fixed_g_energy_feasibility(
            y, g=0.2, problem="DTLZ6", k=4)
        assert outcome.observation == "numerically_infeasible"
        assert outcome.numerical_gap < 0


@pytest.mark.parametrize("m", [2, 3, 4, 5, 10])
def test_axis_singular_case_is_feasible(m):
    y = np.zeros(m)
    y[-1] = 1.5
    result = observe_fixed_g_energy_feasibility(
        y, g=.5, problem="DTLZ6", k=4)
    assert result.observation == "numerically_feasible"
    assert result.stopping_objective is None


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
def test_original_evaluator_not_used(monkeypatch, problem):
    import moeabench as mb
    klass = getattr(mb.mops, problem)

    def forbidden(*args, **kwargs):
        raise AssertionError("production GT or evaluator must not run")

    monkeypatch.setattr(klass, "evaluation", forbidden)
    monkeypatch.setattr(klass, "ps", forbidden)
    monkeypatch.setattr(klass, "pf", forbidden)
    y = _original_feasible_point(4, 0.2, pi/4, 0.3)
    assert observe_fixed_g_energy_feasibility(
        y+1e-10, g=0.2, problem=problem, k=4
    ).observation == "numerically_feasible"


@pytest.mark.parametrize("problem,k,g", [
    ("DTLZ5", 4, 1.01), ("DTLZ6", 4, 4.01),
    ("DTLZ2", 4, 0.), ("DTLZ5", 0, 0.),
    ("DTLZ5", 4, -0.1),
])
def test_invalid_problem_or_g_rejected(problem, k, g):
    with pytest.raises(ValueError):
        observe_fixed_g_energy_feasibility(
            np.ones(4), g=g, problem=problem, k=k)


@pytest.mark.parametrize("y", [
    [float("nan"), 1.], [1., -1.], [1.], [[1., 0.]],
])
def test_invalid_caps_rejected(y):
    with pytest.raises(ValueError):
        observe_fixed_g_energy_feasibility(
            y, g=0.0, problem="DTLZ5", k=4)
