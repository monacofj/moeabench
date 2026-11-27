# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New additive strict second-support surface-to-original PS checks."""
from fractions import Fraction
import pytest

from scripts.audit.dtlz56_second_support_surface_actual_source import (
    certify_second_surface_to_original_ps,
    replay_second_surface_to_original_ps,
)
from scripts.audit.dtlz5_patch_actual_source_cover import _decode


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_exact_second_surface_bound_against_feasible_actual_ps(problem):
    if problem=="DTLZ5":
        ps=[(0.,1.,0.,0.),(.125,1.,0.,.5),(.25,1.,0.,1.)]
    else:
        ps=[(0.,1.,0.,0.),(.09375,1.,0.,float(Fraction(1,2)**10)),
            (.1875,1.,0.,1.)]
    result=certify_second_surface_to_original_ps(ps,problem=problem)
    assert replay_second_surface_to_original_ps(
        ps,result,problem=problem)
    assert len(result["grid_to_actual_source_proof_nodes"])==25
    assert result["source_decision_count"]==3
    assert _decode(result["continuous_surface_to_actual_ps_l2_upper"])>0
    assert result["independent_continuous_surface_pf_proof"][
        "second_support_objective"]==2
    assert result["full_positive_g_pf_coverage"]=="unverified"
    assert result["scientific_certification"]=="unverified"


def test_forgery_and_changed_reference_rejected():
    ps=[(0.,1.,0.,.5),(.25,1.,0.,1.)]
    proof=certify_second_surface_to_original_ps(ps,problem="DTLZ5")
    forged=dict(proof)
    forged["continuous_surface_to_actual_ps_l2_upper"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_second_surface_to_original_ps(
        ps,forged,problem="DTLZ5")
    changed=ps.copy()
    changed[0]=(.01,1.,0.,.5)
    assert not replay_second_surface_to_original_ps(
        changed,proof,problem="DTLZ5")


def test_empty_and_wrong_source_rejected():
    with pytest.raises(ValueError):
        certify_second_surface_to_original_ps([],problem="DTLZ5")
    with pytest.raises(ValueError):
        certify_second_surface_to_original_ps(
            [(0.,1.,0.,0.)],problem="DTLZ8")
