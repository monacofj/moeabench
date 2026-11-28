# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive exact source objective-image offray g0 domination tests."""
import pytest
from fractions import Fraction
from scripts.audit.dtlz56_gzero_offray_dominance import (
    certify_gzero_curve_offray_dominance,
    replay_gzero_curve_offray_dominance,
)
from scripts.audit.dtlz56_exact_same_ray import _decode


@pytest.mark.parametrize("problem,tail",[
    ("DTLZ5",Fraction(3,4)),("DTLZ6",Fraction(1,4)),
])
def test_exact_real_offray_gzero_curve_dominates_open_angular(problem,tail):
    # With x2=x3=.5, all inner directions coincide with g=0,
    # and positive-g candidate is strictly farther from the origin.
    point=(Fraction(1,4),Fraction(1,2),Fraction(1,2),tail)
    proof=certify_gzero_curve_offray_dominance(point,problem=problem)
    assert proof["conclusion"]=="rigorously_strictly_dominated_by_gzero_curve"
    assert replay_gzero_curve_offray_dominance(point,proof,problem=problem)
    assert all(_decode(v)>=0 for v in
               proof["squared_coordinate_margins_lower"])
    assert proof["whole_gt_coverage"]=="unverified"


@pytest.mark.parametrize("problem,tail",[
    ("DTLZ5",Fraction(1,2)),("DTLZ6",Fraction(0)),
])
def test_original_gzero_does_not_claim_strict_dominance(problem,tail):
    point=(Fraction(1,4),Fraction(1,2),Fraction(1,2),tail)
    proof=certify_gzero_curve_offray_dominance(point,problem=problem)
    assert proof["conclusion"]=="unverified"
    assert not replay_gzero_curve_offray_dominance(
        point,proof,problem=problem)


@pytest.mark.parametrize("problem,tail",[
    ("DTLZ5",Fraction(3,4)),("DTLZ6",Fraction(1,4)),
])
def test_known_supported_positive_g_point_not_falsely_dominated(problem,tail):
    point=(Fraction(0),Fraction(1),Fraction(0),tail)
    proof=certify_gzero_curve_offray_dominance(point,problem=problem)
    assert proof["conclusion"]=="unverified"
    assert not replay_gzero_curve_offray_dominance(
        point,proof,problem=problem)


def test_forgeries_and_wrong_source_fail_replay():
    x=(Fraction(1,4),Fraction(1,2),Fraction(1,2),Fraction(3,4))
    proof=certify_gzero_curve_offray_dominance(x,problem="DTLZ5")
    altered=dict(proof)
    altered["squared_coordinate_margins_lower"]=[]
    assert not replay_gzero_curve_offray_dominance(
        x,altered,problem="DTLZ5")
    assert not replay_gzero_curve_offray_dominance(
        (Fraction(1,8),*x[1:]),proof,problem="DTLZ5")


@pytest.mark.parametrize("row",[
    (0.,0.,0.),
    (0.,0.,0.,float("nan")),
    (0.,0.,0.,2.),
])
def test_invalid_original_decision_rejected(row):
    with pytest.raises(ValueError):
        certify_gzero_curve_offray_dominance(row,problem="DTLZ5")
