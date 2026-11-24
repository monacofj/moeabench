# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent exact-rational exclusion tests, without regression changes."""
from fractions import Fraction
from math import cos, pi, sin

import numpy as np
import pytest

from scripts.audit.dtlz56_rational_interval import (
    _arctan_rational_bounds, _pi_rational_bounds,
    _sine_lower_bound, rigorously_exclude_g_interval,
)


def _published_objectives(m, g, theta1, inner_value):
    r = 1.0+g
    a = pi/(4*r)
    b = pi/2-a
    angles = [theta1]+[a+inner_value*(b-a)]*(m-2)
    y = np.ones(m)*r
    for j in range(m):
        for i in range(m-j-1):
            y[j] *= cos(angles[i])
        if j:
            y[j] *= sin(angles[m-j-1])
    return y


def test_exact_rational_pi_enclosure_is_positive_and_nested():
    a,b = _pi_rational_bounds(8)
    c,d = _pi_rational_bounds(16)
    assert 3 < a < c < d < b < 4
    assert all(isinstance(v, Fraction) for v in (a,b,c,d))
    # Alternating series widths are exact positive rationals.
    assert b-a > 0
    assert d-c > 0
    assert d-c < b-a
    x,y = _arctan_rational_bounds(5,16)
    assert x < y


@pytest.mark.parametrize("pairs", [1, 2, 4, 8])
def test_exact_sine_lower_bound_is_positive_on_target_domain(pairs):
    pi_lo,_ = _pi_rational_bounds(8)
    x = pi_lo/8
    val = _sine_lower_bound(x,pairs)
    assert isinstance(val,Fraction)
    assert 0 < val < x < 1


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [2,3,4,5,10])
def test_genuine_feasible_interior_point_never_excluded(problem,m):
    g=0.2
    y=_published_objectives(m,g,pi/4,.4)+1e-9
    outcome=rigorously_exclude_g_interval(
        y,g_lower=.1,g_upper=.4,problem=problem,k=4
    )
    assert outcome.conclusion=="unresolved",outcome.to_dict()
    assert outcome.to_dict()["scientific_certification"]=="unverified"


@pytest.mark.parametrize("m", [2,3,4,5,10])
def test_rational_certificate_excludes_clearly_infeasible_interval(m):
    outcome=rigorously_exclude_g_interval(
        np.zeros(m),g_lower=.1,g_upper=.5,
        problem="DTLZ6",k=4
    )
    assert outcome.conclusion=="rigorously_excluded"
    assert outcome.stopping_objective is not None
    assert Fraction(outcome.strict_margin)>0
    assert outcome.to_dict()["scientific_certification"]=="unverified"


def test_axis_feasible_at_interior_g_does_not_get_excluded():
    y=np.array([0.,0.,0.,1.5])
    result=rigorously_exclude_g_interval(
        y,g_lower=.3,g_upper=.7,problem="DTLZ6",k=4
    )
    assert result.conclusion=="unresolved"


def test_singleton_interval_with_feasible_point_is_not_excluded():
    g=.5
    y=_published_objectives(5,g,0.4,.6)+1e-8
    result=rigorously_exclude_g_interval(
        y,g_lower=g,g_upper=g,problem="DTLZ5",k=4
    )
    assert result.conclusion=="unresolved"


def test_rational_result_does_not_depend_on_float_sin(monkeypatch):
    import math
    def forbidden(*args):
        raise AssertionError("hardware sine must not be called")
    monkeypatch.setattr(math,"sin",forbidden)
    a=rigorously_exclude_g_interval(
        [0.,0.,0.,0.],g_lower=0.1,g_upper=0.2,
        problem="DTLZ6",k=4
    )
    assert a.conclusion=="rigorously_excluded"


@pytest.mark.parametrize("problem,k,lo,hi", [
    ("DTLZ1",4,0.,0.1), ("DTLZ5",0,0.,0.1),
    ("DTLZ5",4,-0.1,0.2), ("DTLZ5",4,0.2,0.1),
    ("DTLZ5",4,0.2,1.01), ("DTLZ6",4,0.2,4.01),
])
def test_reject_unsupported_problem_or_g(problem,k,lo,hi):
    with pytest.raises(ValueError):
        rigorously_exclude_g_interval(
            [1.,1.,1.],g_lower=lo,g_upper=hi,problem=problem,k=k
        )


@pytest.mark.parametrize("caps", [
    [float("nan"),1.], [-1.,1.], [0.], [[1.,0.],[0.,1.]],
])
def test_reject_bad_caps(caps):
    with pytest.raises(ValueError):
        rigorously_exclude_g_interval(
            caps,g_lower=0.,g_upper=.2,problem="DTLZ5",k=4
        )


def test_unsupported_precision_rejected():
    with pytest.raises(ValueError):
        rigorously_exclude_g_interval(
            [0.,0.],g_lower=0.,g_upper=.1,
            problem="DTLZ5",k=4,pi_terms=100
        )
