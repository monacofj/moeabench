# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent mathematical checks for #89; no old test modified."""

from math import factorial, gamma, pi

import pytest

from scripts.audit.hv_analytic_dtlz_issue89 import continuous_hv, sample_check


@pytest.mark.parametrize("objectives", [2, 3, 4, 5, 8])
def test_closed_form_exact_dtlz1_hv(objectives):
    assert continuous_hv("DTLZ1", objectives) == pytest.approx(
        1.1**objectives - 1/factorial(objectives), abs=1e-12
    )


@pytest.mark.parametrize("objectives", [2, 3, 4, 5, 8])
def test_closed_form_exact_dtlz2_hv(objectives):
    assert continuous_hv("DTLZ2", objectives) == pytest.approx(
        1.1**objectives
        - pi**(objectives/2)/(2**objectives*gamma(objectives/2+1)),
        abs=1e-12
    )


def test_known_two_objective_closed_forms():
    assert continuous_hv("DTLZ1", 2) == pytest.approx(.71)
    assert continuous_hv("DTLZ2", 2) == pytest.approx(1.21-pi/4)


@pytest.mark.parametrize("problem,objectives", [
    ("DTLZ1", 2), ("DTLZ1", 3), ("DTLZ1", 4),
    ("DTLZ2", 2), ("DTLZ2", 3), ("DTLZ2", 4),
])
def test_actual_pf_samples_below_proved_continuous_hv(problem, objectives):
    evidence = sample_check(problem, objectives, points=64)
    assert 0 < evidence["finite_to_exact_ratio"] < 1


@pytest.mark.parametrize("invalid", [0, 1, 1.5, "3"])
def test_reject_unproved_objective_dimensions(invalid):
    with pytest.raises(ValueError, match="integer"):
        continuous_hv("DTLZ1", invalid)


def test_reject_nonproven_reference_and_problem():
    with pytest.raises(ValueError, match="r >= 1"):
        continuous_hv("DTLZ2", 3, reference=.9)
    with pytest.raises(ValueError, match="Only DTLZ1"):
        continuous_hv("DTLZ5", 3)
