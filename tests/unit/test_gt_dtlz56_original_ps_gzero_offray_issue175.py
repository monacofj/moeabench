# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive original-source selected offray g0 PF screen tests."""
import pytest
from fractions import Fraction
from scripts.audit.dtlz56_original_ps_gzero_offray import (
    certify_original_ps_gzero_offray,replay_original_ps_gzero_offray,
)


@pytest.mark.parametrize("problem,tail,gzero_tail",[
    ("DTLZ5",.75,.5),("DTLZ6",.25,0.),
])
def test_source_proved_dominated_and_all_other_rows_abstain(problem,tail,gzero_tail):
    ps=[
        (.25,.5,.5,tail),   # exact g0 PF-curve dominator
        (0.,1.,0.,tail),    # already known exact global positive-g PF
        (.5,.5,.5,gzero_tail),  # complete g0 PF
    ]
    p=certify_original_ps_gzero_offray(ps,problem=problem)
    assert replay_original_ps_gzero_offray(ps,p,problem=problem)
    assert p["rigorously_gzero_offray_dominated_indices"]==[0]
    assert p["whole_gt_membership"]=="rejected_has_dominated_member"
    assert p["rows"][1]["certificate"]["conclusion"]=="unverified"
    assert p["rows"][2]["certificate"]["conclusion"]=="unverified"


def test_forged_and_altered_source_does_not_replay():
    ps=[(.25,.5,.5,.75)]
    p=certify_original_ps_gzero_offray(ps,problem="DTLZ5")
    altered=dict(p)
    altered["rigorously_gzero_offray_dominated_indices"]=[]
    assert not replay_original_ps_gzero_offray(
        ps,altered,problem="DTLZ5")
    assert not replay_original_ps_gzero_offray(
        [(.125,.5,.5,.75)],p,problem="DTLZ5")


@pytest.mark.parametrize("ps",[[],[(0.,.5,.5)],[(0.,.5,.5,float("nan"))]])
def test_invalid_source_fails_closed(ps):
    with pytest.raises(ValueError):
        certify_original_ps_gzero_offray(ps,problem="DTLZ5")
