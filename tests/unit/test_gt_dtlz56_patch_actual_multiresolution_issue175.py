# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive finite-budget test of rigorous nested-grid source bounds."""
from fractions import Fraction

import pytest

from scripts.audit.dtlz56_patch_actual_multiresolution import (
    multiresolution_source_certificate,
    replay_multiresolution_source_certificate,
)
from scripts.audit.dtlz5_patch_actual_source_cover import _decode


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_nested_grids_replay_independent_real_decision_images(problem):
    if problem=="DTLZ5":
        source=[(x,1.,1.,t)
                for x in (0.,1/8) for t in (.5,.75)]
    else:
        source=[(x,1.,1.,float(Fraction(u)**10))
                for x in (0.,1/8) for u in (0.,1/4)]
    record=multiresolution_source_certificate(source,problem=problem)
    assert replay_multiresolution_source_certificate(
        source,record,problem=problem)
    assert [r["grid"] for r in record["records"]]==["3x3","5x5"]
    bounds=[_decode(r["patch_to_selected_source_l2_upper"])
            for r in record["records"]]
    assert _decode(record["best_rigorous_upper"])==min(bounds)
    for r in record["records"]:
        assert _decode(r["patch_to_selected_source_l2_upper"])==(
            _decode(r["continuum_to_grid_l2_upper"])+
            _decode(r["grid_to_selected_source_l1_upper"]))
    assert record["scientific_certification"]=="unverified"
    assert record["indicator_fitness"]=="unverified"


def test_forged_margin_never_replays():
    source=[(0.,1.,1.,.5),(1/8,1.,1.,.75)]
    record=multiresolution_source_certificate(
        source,problem="DTLZ5")
    forged=dict(record)
    forged["best_rigorous_upper"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_multiresolution_source_certificate(
        source,forged,problem="DTLZ5")


def test_unsupported_problem_fails_closed():
    with pytest.raises(ValueError):
        multiresolution_source_certificate(
            [(0.,1.,1.,0.)],problem="DTLZ7")
