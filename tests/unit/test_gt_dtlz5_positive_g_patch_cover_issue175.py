# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New additive rational continuous positive-g patch theorem tests."""
from fractions import Fraction

import pytest

from scripts.audit.dtlz5_positive_g_patch_cover import (
    certify_positive_g_patch_cover,replay_positive_g_patch_cover,
)


def _fraction(field):
    return Fraction(int(field["numerator_hex"],16),
                    int(field["denominator_hex"],16))


def test_uniform_positive_g_patch_and_grid_all_exact_pf():
    cert=certify_positive_g_patch_cover()
    assert cert["conclusion"]=="rigorously_bounded_continuous_pf_patch"
    assert replay_positive_g_patch_cover(cert)
    assert len(cert["grid_decision_images"])==9
    assert len([x for x in cert["grid_decision_images"]
                if x["global_pf_membership_proof"]!="DTLZ56-UNIT-RADIUS"])==6
    assert _fraction(cert["uniform_pf_strict_margin"])>0
    upper=_fraction(cert["raw_euclidean_continuous_patch_cover_upper"])
    assert Fraction(1,5)<upper<Fraction(1,4)
    assert cert["whole_gt_coverage"]=="unverified"
    assert cert["indicator_fitness"]=="unverified"


def test_finer_exact_grid_decreases_continuous_cover_upper():
    coarse=certify_positive_g_patch_cover()
    refined=certify_positive_g_patch_cover(
        x1_grid=(0.,1/32,1/16,3/32,1/8),
        tail_grid=(.5,9/16,5/8,11/16,3/4))
    assert _fraction(
        refined["raw_euclidean_continuous_patch_cover_upper"]
    ) < _fraction(coarse["raw_euclidean_continuous_patch_cover_upper"])
    assert replay_positive_g_patch_cover(
        refined,
        x1_grid=(0.,1/32,1/16,3/32,1/8),
        tail_grid=(.5,9/16,5/8,11/16,3/4))


def test_forged_bound_not_replayable():
    cert=certify_positive_g_patch_cover()
    bogus=dict(cert)
    bogus["raw_euclidean_continuous_patch_cover_upper"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_positive_g_patch_cover(bogus)
    bogus=dict(cert)
    bogus["grid_decision_images"]=[]
    assert not replay_positive_g_patch_cover(bogus)


@pytest.mark.parametrize("axis",[
    (0.,1/16), (0.,1/16,1/16,1/8),
    (0.,1/8,1/16), (0.,float("nan"),1/8),
    (-.1,0.,1/8), (0.,1/16,.25),
])
def test_invalid_x_axis_rejected(axis):
    with pytest.raises(ValueError):
        certify_positive_g_patch_cover(x1_grid=axis)


def test_invalid_tail_or_pi_rejected():
    with pytest.raises(ValueError):
        certify_positive_g_patch_cover(tail_grid=(.5,.75,.625))
    with pytest.raises(ValueError):
        certify_positive_g_patch_cover(pi_terms=0)
