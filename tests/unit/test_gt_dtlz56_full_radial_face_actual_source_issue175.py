# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New source-bound exact complete-radial-face covering replay tests."""
from fractions import Fraction
import pytest

from scripts.audit.dtlz56_full_radial_face_actual_source import (
    certify_full_radial_face_to_source,replay_full_radial_face_to_source,
)
from scripts.audit.dtlz5_patch_actual_source_cover import _decode


@pytest.mark.parametrize("problem",("DTLZ5","DTLZ6"))
def test_source_bound_full_radial_face_proof_replays(problem):
    if problem=="DTLZ5":
        source=[(0.,1.,1.,0.),(.125,1.,1.,.5),(.25,1.,1.,1.)]
    else:
        source=[(0.,1.,1.,0.),(.09375,1.,1.,float(Fraction(1,2)**10)),
                (.1875,1.,1.,1.)]
    record=certify_full_radial_face_to_source(
        source,problem=problem)
    assert replay_full_radial_face_to_source(
        source,record,problem=problem)
    assert record["source_decision_count"]==3
    assert len(record["grid_nearest_actual_source"])==25
    assert _decode(record["full_radial_face_to_actual_source_l2_upper"])>0
    assert record["whole_pf_coverage"]=="unverified"
    assert record["scientific_certification"]=="unverified"


def test_different_source_or_forged_upper_is_never_replayed():
    source=[(0.,1.,1.,.5),(0.25,1.,1.,1.)]
    proof=certify_full_radial_face_to_source(
        source,problem="DTLZ5")
    bogus=dict(proof)
    bogus["full_radial_face_to_actual_source_l2_upper"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_full_radial_face_to_source(
        source,bogus,problem="DTLZ5")
    changed=source.copy()
    changed[0]=(0.01,1.,1.,.5)
    assert not replay_full_radial_face_to_source(
        changed,proof,problem="DTLZ5")


def test_invalid_ps_and_wrong_problem_fail_closed():
    with pytest.raises(ValueError):
        certify_full_radial_face_to_source(
            [],problem="DTLZ5")
    with pytest.raises(ValueError):
        certify_full_radial_face_to_source(
            [(0.,1.,1.,0.)],problem="DTLZ8")
