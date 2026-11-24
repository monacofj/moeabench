# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive exact curve-dominance and replay tests; no regression edits."""
from copy import deepcopy
from fractions import Fraction

import pytest

from scripts.audit.dtlz56_exact_g0_curve import (
    exact_g0_curve_dominance, replay_exact_g0_curve_dominance,
)


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [2, 3, 4, 5, 10])
def test_last_axis_curve_dominates_positive_g_at_any_dimension(problem, m):
    # Published original feasible theta_1=pi/2 and g=1/4.
    y = [Fraction(0)]*(m-1)+[Fraction(5,4)]
    result = exact_g0_curve_dominance(y, problem=problem, k=4)
    assert result.conclusion == "exact_g0_curve_dominator"
    assert replay_exact_g0_curve_dominance(
        y, result.to_dict(), problem=problem, k=4)


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [3, 4, 5, 10])
def test_different_ray_dominator_on_original_angular_boundary(problem, m):
    # Rational objective caps, without assuming they form a feasible GT row.
    # This tests the curve-cap predicate, not original image membership.
    y = [Fraction(1)]*m
    result = exact_g0_curve_dominance(y, problem=problem, k=4)
    assert result.conclusion == "exact_g0_curve_dominator"


@pytest.mark.parametrize("m", [2,3,4,5,10])
def test_exact_unit_radius_endpoint_is_not_strict(m):
    y = [Fraction(0)]*(m-1)+[Fraction(1)]
    result = exact_g0_curve_dominance(
        y, problem="DTLZ5", k=4)
    assert result.conclusion == "weak_g0_curve_fit_unit_radius"


def test_exact_curve_failure_is_not_global_nondominance():
    # Nonnegative, norm>1, but the first objective is too small to
    # admit any unit-radius point on the complete curve.
    y = [Fraction(0),Fraction(2),Fraction(2,3)]
    result = exact_g0_curve_dominance(
        y, problem="DTLZ6", k=4)
    assert result.conclusion == "no_g0_curve_dominator"
    assert result.obstructing_objective == 1
    assert result.to_dict()["scientific_certification"] == "unverified"


def test_exact_dyadic_boundary_is_accepted_without_tolerance():
    # M=4, y_M=0: g0 t=0 has squared coordinates 1/4, 1/4, 1/2.
    # Construct exact caps above by rational choices avoiding sqrt(2).
    y = [Fraction(1,2),Fraction(1,2),Fraction(3,4),Fraction(0)]
    result = exact_g0_curve_dominance(
        y, problem="DTLZ5", k=4)
    assert result.conclusion == "exact_g0_curve_dominator"


def test_very_small_strict_excess_is_proved_without_numeric_gap():
    delta = Fraction(1,2**220)
    y = [Fraction(0),Fraction(0),Fraction(1)+delta]
    result = exact_g0_curve_dominance(
        y, problem="DTLZ6", k=4)
    assert result.conclusion == "exact_g0_curve_dominator"


def test_tampered_certificate_fails_independent_replay():
    y = [Fraction(1,2),Fraction(1,2),Fraction(3,4),Fraction(0)]
    record = exact_g0_curve_dominance(
        y, problem="DTLZ5", k=4).to_dict()
    assert replay_exact_g0_curve_dominance(
        y, record, problem="DTLZ5", k=4)
    forged = deepcopy(record)
    forged["radius_squared"]["numerator_hex"] = "0x0"
    assert not replay_exact_g0_curve_dominance(
        y, forged, problem="DTLZ5", k=4)


@pytest.mark.parametrize("y,problem,k", [
    ([1.], "DTLZ5", 4),
    ([Fraction(-1),Fraction(1)], "DTLZ5", 4),
    ([float("nan"),1], "DTLZ6", 4),
    ([1,1], "DTLZ2", 4),
    ([1,1], "DTLZ6", 0),
])
def test_reject_invalid_input(y,problem,k):
    with pytest.raises(ValueError):
        exact_g0_curve_dominance(y, problem=problem,k=k)
