# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive tests for a tighter exact rational original DTLZ5 PF theorem."""
from copy import deepcopy
from fractions import Fraction as Q
import json

import numpy as np
import pytest

from scripts.audit.dtlz56_paired_gt_evidence import _original_objectives
from scripts.audit.dtlz56_positive_g_open_first_angle import (
    certify_open_first_angle_global_pf,
)
from scripts.audit.dtlz56_tight_sine_global_pf import (
    _taylor_sine_upper,
    certify_tight_sine_global_pf,
    replay_tight_sine_global_pf,
    audit_tight_sine_original_rows,
)


# Source-bound row coordinates captured after the PR #210 face repair,
# 2026-11-24 original DTLZ5 M4 K1 full maximin selected n16 sample.
# These are exactly defined binary floats, NOT rounded proof inputs.
REPRODUCED = (
    (0.3108514266364233, 1.0, 0.0, 1.0),
    (0.3104519947719909, 1.0, 1.0, 1.0),
)


@pytest.mark.parametrize("source_row", REPRODUCED)
def test_new_exact_certificate_closes_two_real_original_dtlz5_gaps(source_row):
    positions=source_row[:3]
    g=(Q.from_float(source_row[-1])-Q(1,2))**2
    assert g==Q(1,4)
    old=certify_open_first_angle_global_pf(
        positions,g=g,problem="DTLZ5",k=1)
    assert old.conclusion=="unverified"
    new=certify_tight_sine_global_pf(
        positions,g=g,problem="DTLZ5",k=1)
    assert new.conclusion=="rigorously_global_pf_member"
    assert new.supporting_objective in (1,2)
    assert new.sine_power==2
    assert replay_tight_sine_global_pf(
        positions,new.to_dict(),g=g,problem="DTLZ5",k=1)
    json.dumps(new.to_dict(),allow_nan=False)
    assert new.to_dict()["scientific_certification"]=="unverified"


def test_third_source_unresolved_position_must_not_be_falsely_certified():
    positions=(0.1671198950208142,0.999999658561122,1.0)
    g=(Q.from_float(0.6805481014263987)-Q(1,2))**2
    new=certify_tight_sine_global_pf(
        positions,g=g,problem="DTLZ5",k=1)
    assert new.conclusion=="unverified"
    assert new.supporting_objective is None
    assert not replay_tight_sine_global_pf(
        positions,new,g=g,problem="DTLZ5",k=1)


@pytest.mark.parametrize("m", [4,5,7,10])
@pytest.mark.parametrize("problem", ["DTLZ5","DTLZ6"])
def test_sufficient_strata_multiple_dimensions_and_both_original_problems(
    m,problem,
):
    positions=[Q(5,16),Q(1),Q(0)]+[Q(1,3)]*(m-4)
    result=certify_tight_sine_global_pf(
        positions,g=Q(1,4),problem=problem,k=1)
    assert result.conclusion=="rigorously_global_pf_member"
    assert result.sine_power==2


def test_sharp_sine_upper_is_still_rigorously_one_sided():
    from scripts.audit.dtlz56_rational_interval import _pi_rational_bounds
    _,pi_hi=_pi_rational_bounds(8)
    for u in (Q(0),Q(1,16),Q(5,16),Q(1,2),Q(1)):
        theta_hi=pi_hi*u/2
        upper=_taylor_sine_upper(theta_hi)
        assert upper >= 0
        assert float(upper)+1e-14 >= np.sin(np.pi*float(u)/2)
        assert upper <= theta_hi


def test_replay_rejects_forged_margin_and_original_decision():
    positions=[Q(5,16),Q(1),Q(0)]
    g=Q(1,4)
    original=certify_tight_sine_global_pf(
        positions,g=g,problem="DTLZ5",k=1).to_dict()
    assert replay_tight_sine_global_pf(
        positions,original,g=g,problem="DTLZ5",k=1)
    forged=deepcopy(original)
    forged["strict_margin"]["numerator_hex"]="0x0"
    assert not replay_tight_sine_global_pf(
        positions,forged,g=g,problem="DTLZ5",k=1)
    assert not replay_tight_sine_global_pf(
        [Q(1,4),Q(1),Q(0)],original,g=g,problem="DTLZ5",k=1)


def test_companion_ledger_captures_two_real_rows_and_abstains_on_third():
    rows=np.asarray([*REPRODUCED,
        (0.1671198950208142,0.999999658561122,1.0,
         0.6805481014263987)],dtype=float)
    objectives,_=_original_objectives(
        rows,problem="DTLZ5",m=4,k=1)
    result=audit_tight_sine_original_rows(
        rows,objectives,problem="DTLZ5",m=4,k=1)
    assert result["proved_global_decision_image_indices"]==[0,1]
    assert result["proved_global_decision_image_count"]==2
    assert result["rows"][-1]["status"]=="unverified"
    assert result["scientific_certification"]=="unverified"
    wrong=objectives.copy()
    wrong[0,0]+=0.01
    rejected=audit_tight_sine_original_rows(
        rows,wrong,problem="DTLZ5",m=4,k=1)
    assert rejected["proved_global_decision_image_indices"]==[1]


@pytest.mark.parametrize("position,g,problem,k",[
    ([Q(0),Q(1)],Q(1,4),"DTLZ5",1),
    ([Q(-1),Q(1),Q(0)],Q(1,4),"DTLZ5",1),
    ([Q(1,2),Q(1),Q(0)],Q(0),"DTLZ5",1),
    ([Q(1,2),Q(1),Q(0)],Q(1),"DTLZ5",1),
    ([Q(1,2),Q(1),Q(0)],Q(1,4),"DTLZ1",1),
    ([Q(1,2),Q(1),Q(0)],Q(1,4),"DTLZ5",0),
])
def test_invalid_inputs_rejected(position,g,problem,k):
    with pytest.raises(ValueError):
        certify_tight_sine_global_pf(
            position,g=g,problem=problem,k=k)


def test_no_sampler_or_production_evaluator_required(monkeypatch):
    from moeabench.mops import DTLZ5
    def forbidden(*a,**kw):
        raise AssertionError("No production GT computation is permitted")
    monkeypatch.setattr(DTLZ5,"ps",forbidden)
    monkeypatch.setattr(DTLZ5,"evaluation",forbidden)
    result=certify_tight_sine_global_pf(
        [Q(5,16),Q(1),Q(0)],g=Q(1,4),
        problem="DTLZ5",k=1,
    )
    assert result.conclusion=="rigorously_global_pf_member"
