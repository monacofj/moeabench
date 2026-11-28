# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive exact continuous g=0 component covering proof tests."""
from fractions import Fraction

import pytest

from scripts.audit.dtlz56_gzero_continuous_cover import (
    certify_gzero_curve_cover, replay_gzero_curve_cover,
)


def anchor_rows(problem, xs=(0.,1./3,2./3,1.)):
    tail=.5 if problem=="DTLZ5" else 0.
    return [(x,.1,.9,tail) for x in xs]


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_exact_cover_proof_replays_and_preserves_scope(problem):
    rows=anchor_rows(problem)
    cert=certify_gzero_curve_cover(rows,problem=problem)
    assert cert["conclusion"]=="rigorously_bounded_continuous_gzero_cover"
    assert replay_gzero_curve_cover(rows,cert,problem=problem)
    bound=cert["raw_euclidean_cover_upper"]
    upper=Fraction(int(bound["numerator_hex"],16),
                   int(bound["denominator_hex"],16))
    assert Fraction(1,4)<upper<Fraction(27,100)
    mean=cert["uniform_x1_mean_raw_distance_upper"]
    mean_upper=Fraction(int(mean["numerator_hex"],16),
                        int(mean["denominator_hex"],16))
    assert Fraction(12,100)<mean_upper<Fraction(14,100)
    assert mean_upper < upper
    assert cert["continuous_gzero_component_only"] is True
    assert cert["whole_gt_coverage"]=="unverified"
    assert cert["scientific_certification"]=="unverified"


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_missing_curve_endpoint_fails_closed(problem):
    rows=anchor_rows(problem,xs=(0.,1./3,2./3))
    cert=certify_gzero_curve_cover(rows,problem=problem)
    assert cert["conclusion"]=="unverified_missing_curve_endpoint"
    assert cert["uniform_x1_mean_raw_distance_upper"] is None
    assert not replay_gzero_curve_cover(rows,cert,problem=problem)


def test_stricter_anchor_mesh_has_no_larger_bound():
    coarse=certify_gzero_curve_cover(
        anchor_rows("DTLZ5",xs=(0.,.5,1.)),problem="DTLZ5")
    fine=certify_gzero_curve_cover(
        anchor_rows("DTLZ5",xs=(0.,.25,.5,.75,1.)),problem="DTLZ5")
    def rational(record):
        x=record["raw_euclidean_cover_upper"]
        return Fraction(int(x["numerator_hex"],16),
                        int(x["denominator_hex"],16))
    assert rational(fine)<rational(coarse)
    def mean(record):
        x=record["uniform_x1_mean_raw_distance_upper"]
        return Fraction(int(x["numerator_hex"],16),
                        int(x["denominator_hex"],16))
    assert mean(fine)<mean(coarse)


def test_forged_bound_and_wrong_problem_cannot_replay():
    rows=anchor_rows("DTLZ6")
    proof=certify_gzero_curve_cover(rows,problem="DTLZ6")
    bogus=dict(proof)
    bogus["raw_euclidean_cover_upper"]={"numerator_hex":"0x0",
                                        "denominator_hex":"0x1"}
    assert not replay_gzero_curve_cover(rows,bogus,problem="DTLZ6")
    assert not replay_gzero_curve_cover(rows,proof,problem="DTLZ5")


@pytest.mark.parametrize("bad",[
    [(0.,.2,.2)],
    [(0.,.2,.2,-1.)],
    [(0.,.2,.2,float("nan"))],
])
def test_invalid_decisions_rejected(bad):
    with pytest.raises(ValueError):
        certify_gzero_curve_cover(bad,problem="DTLZ5")
