# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive conservative second-surface union verification tests."""
import pytest
from scripts.audit.dtlz56_two_surface_union_cover import (
    certify_two_surface_union,replay_two_surface_union,
)
from scripts.audit.dtlz56_full_radial_face_actual_source import (
    certify_full_radial_face_to_source,
)
from scripts.audit.dtlz56_second_support_actual_source import (
    certify_second_support_to_actual_ps,
)
from scripts.audit.dtlz56_second_support_surface_actual_source import (
    certify_second_surface_to_original_ps,
)
from scripts.audit.dtlz5_patch_actual_source_cover import _decode


@pytest.fixture(scope="module")
def original_source_proof_triplet():
    ps=[(x,.5,.5,.5) for x in (0.,1/3,2/3,1.)]
    return (
        ps,
        certify_full_radial_face_to_source(ps,problem="DTLZ5"),
        certify_second_support_to_actual_ps(ps,problem="DTLZ5"),
        certify_second_surface_to_original_ps(ps,problem="DTLZ5"),
    )


def test_proved_union_uses_stronger_second_support_surface(original_source_proof_triplet):
    ps,first,curve,surface=original_source_proof_triplet
    result=certify_two_surface_union(
        ps,problem="DTLZ5",first_report=first,
        second_curve_report=curve,second_surface_report=surface)
    assert replay_two_surface_union(
        ps,result,problem="DTLZ5",first_report=first,
        second_curve_report=curve,second_surface_report=surface)
    bounds=[_decode(x) for x in result["component_bounds"].values()]
    assert len(bounds)==3
    assert _decode(result["union_continuous_raw_l2_upper"])==max(bounds)
    assert result["second_support_radial_curve_included_in_second_surface"]
    assert result["complete_original_pf_decomposition"]=="unverified"
    assert result["whole_gt_coverage"]=="unverified"


def test_changed_ps_and_forged_upper_are_rejected(original_source_proof_triplet):
    ps,first,curve,surface=original_source_proof_triplet
    result=certify_two_surface_union(
        ps,problem="DTLZ5",first_report=first,
        second_curve_report=curve,second_surface_report=surface)
    forged=dict(result)
    forged["union_continuous_raw_l2_upper"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_two_surface_union(
        ps,forged,problem="DTLZ5",first_report=first,
        second_curve_report=curve,second_surface_report=surface)
    altered=ps.copy()
    altered[0]=(.01,.5,.5,.5)
    assert not replay_two_surface_union(
        altered,result,problem="DTLZ5",first_report=first,
        second_curve_report=curve,second_surface_report=surface)
