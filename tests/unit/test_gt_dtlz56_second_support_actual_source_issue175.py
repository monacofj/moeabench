# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent actual-PS second-support curve covering tests."""
import pytest

from scripts.audit.dtlz56_second_support_actual_source import (
    certify_second_support_to_actual_ps,
    replay_second_support_to_actual_ps,
)
from scripts.audit.dtlz5_patch_actual_source_cover import _decode


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_rigorous_complete_second_support_to_actual_selected_ps(problem):
    if problem=="DTLZ5":
        ps=[(0.,1.,0.,0.),(0.,1.,0.,.5),(0.,1.,0.,1.)]
    else:
        ps=[(0.,1.,0.,0.),(0.,1.,0.,float((.5)**10)),
            (0.,1.,0.,1.)]
    result=certify_second_support_to_actual_ps(ps,problem=problem)
    assert replay_second_support_to_actual_ps(ps,result,problem=problem)
    assert len(result["nearest_source_grid_nodes"])==9
    assert result["selected_decision_count"]==3
    assert _decode(result["continuous_curve_to_actual_source_l2_upper"])>0
    assert result["whole_pf_coverage"]=="unverified"
    assert result["finite_F_alignment"]=="unverified"
    assert result["scientific_certification"]=="unverified"


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_source_modification_and_forgery_rejected(problem):
    ps=[(0.,1.,0.,0.),(0.,1.,0.,1.)]
    proof=certify_second_support_to_actual_ps(ps,problem=problem)
    bogus=dict(proof)
    bogus["continuous_curve_to_actual_source_l2_upper"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_second_support_to_actual_ps(ps,bogus,problem=problem)
    changed=ps.copy()
    changed[0]=(0.01,1.,0.,0.)
    assert not replay_second_support_to_actual_ps(
        changed,proof,problem=problem)


def test_empty_or_wrong_source_fails_closed():
    with pytest.raises(ValueError):
        certify_second_support_to_actual_ps([],problem="DTLZ5")
    with pytest.raises(ValueError):
        certify_second_support_to_actual_ps(
            [(0.,1.,0.,0.)],problem="DTLZ8")
