# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additional rigorous irrational-DTLZ6 radial support theorem tests.

No historical regression test is changed; old symbolic triage still
abstains when g is irrational. This is a separately named extension.
"""
from copy import deepcopy
from fractions import Fraction as Q
from math import cos, pi, sin

import numpy as np
import pytest

from scripts.audit.dtlz56_dtlz6_radial_enclosure import (
    rational_tenth_root_enclosure,
    certify_dtlz6_radial_interval_pf,
    replay_dtlz6_radial_interval_pf,
    audit_dtlz6_irrational_radial_rows,
)
from scripts.audit.dtlz56_paired_gt_evidence import _original_objectives
from scripts.audit.dtlz56_paired_symbolic_audit import (
    audit_symbolic_candidate,
)


@pytest.mark.parametrize("tail", [
    Q(0), Q(1), Q(1,1024), Q(1,2), Q(1,3),
    Q(1,2**51), Q((2**53)-1,2**53),
])
@pytest.mark.parametrize("bits", [4, 20, 40])
def test_exact_integer_tenth_power_enclosures(tail,bits):
    lo,hi=rational_tenth_root_enclosure(tail,bits=bits)
    assert 0<=lo<=hi<=1
    assert lo**10<=tail<=hi**10
    assert hi-lo<=Q(1,2**bits)
    assert (lo==hi)==(lo**10==tail)


@pytest.mark.parametrize("m", [4,5,7,10])
@pytest.mark.parametrize("first", [Q(0),Q(1,32),Q(1,16)])
def test_irrational_g_globally_pareto_optimal_boundary(m,first):
    # At least one positive irrational tenth root in exact arithmetic.
    u=[first,Q(1),Q(0)]+[Q(1,3)]*(m-4)
    tail=[Q(1,2)]
    result=certify_dtlz6_radial_interval_pf(u,tail,k=1)
    assert result.conclusion=="rigorously_global_pf_member"
    assert result.supporting_objective==m-2
    assert result.sine_power==2
    assert result.to_dict()["scientific_certification"]=="unverified"
    assert replay_dtlz6_radial_interval_pf(
        u,tail,result.to_dict(),k=1)
    lower=Q(int(result.g_lower["numerator_hex"],16),
            int(result.g_lower["denominator_hex"],16))
    upper=Q(int(result.g_upper["numerator_hex"],16),
            int(result.g_upper["denominator_hex"],16))
    assert lower**10<=tail[0]<=upper**10
    assert lower<upper  # g is not a rational tenth root


def test_small_nonzero_tail_does_not_need_positive_lower_bound():
    u=[Q(0),Q(1),Q(0)]
    tail=[Q(1,2**1000)]
    result=certify_dtlz6_radial_interval_pf(
        u,tail,k=1,root_bits=8)
    assert result.conclusion=="rigorously_global_pf_member"


@pytest.mark.parametrize("first", [Q(1,4),Q(1,2),Q(1)])
def test_unsafe_first_angle_must_abstain(first):
    result=certify_dtlz6_radial_interval_pf(
        [first,Q(1),Q(0)],[Q(1,2)],k=1)
    assert result.conclusion=="unverified"
    assert not replay_dtlz6_radial_interval_pf(
        [first,Q(1),Q(0)],[Q(1,2)],result,k=1)


def test_off_support_and_zero_radial_abstain():
    for positions,tail in [
        ([Q(1,16),Q(1,2),Q(1,2)],[Q(1,2)]),
        ([Q(1,16),Q(1),Q(0)],[Q(0)]),
    ]:
        result=certify_dtlz6_radial_interval_pf(
            positions,tail,k=1)
        assert result.conclusion=="unverified"


def test_lossless_hex_certificate_forgery_rejected():
    positions=[Q(1,16),Q(1),Q(0)]
    tail=[Q(1,2)]
    original=certify_dtlz6_radial_interval_pf(
        positions,tail,k=1).to_dict()
    assert replay_dtlz6_radial_interval_pf(
        positions,tail,original,k=1)
    for key in ("g_upper","strict_margin"):
        altered=deepcopy(original)
        altered[key]["numerator_hex"]="0x0"
        assert not replay_dtlz6_radial_interval_pf(
            positions,tail,altered,k=1)
    assert not replay_dtlz6_radial_interval_pf(
        positions,[Q(1,3)],original,k=1)


def test_opt_in_audits_original_real_decisions_not_historical_classifier():
    m,k=4,1
    x=np.array([
        [0.,1.,0.,.5],
        [1/16,1.,0.,.5],
        [.5,1.,0.,.5],
        [0.,.5,.5,.5],
    ])
    y,_=_original_objectives(x,problem="DTLZ6",m=m,k=k)
    historical=audit_symbolic_candidate(x,y,problem="DTLZ6",m=m,k=k)
    assert historical["counts"]["unverified"]==4
    upgraded=audit_dtlz6_irrational_radial_rows(
        x,y,m=m,k=k)
    assert upgraded["counts"]=={
        "symbolic_decision_global_pf":2,
        "unverified":2,
    }
    assert upgraded["whole_reference_membership"]=="unverified"
    assert upgraded["scientific_certification"]=="unverified"
    assert upgraded["rows"][0]["certificate"]["conclusion"]=="rigorously_global_pf_member"
    corrupted=y.copy()
    corrupted[1,0]+=0.01
    bad=audit_dtlz6_irrational_radial_rows(
        x,corrupted,m=m,k=k,tolerance=1e-10)
    assert bad["counts"]["symbolic_decision_global_pf"]==1
    assert bad["source_objective_numerical_alignment"]=="fail"


@pytest.mark.parametrize("positions,tail,k,bits", [
    ([Q(0),Q(1)], [Q(1,2)], 1, 40),
    ([Q(-1),Q(1),Q(0)], [Q(1,2)], 1, 40),
    ([Q(0),Q(1),Q(0)], [Q(-1,2)], 1, 40),
    ([Q(0),Q(1),Q(0)], [Q(1,2),Q(1,2)], 1, 40),
    ([Q(0),Q(1),Q(0)], [Q(1,2)], 0, 40),
    ([Q(0),Q(1),Q(0)], [Q(1,2)], 1, 0),
])
def test_invalid_inputs_rejected(positions,tail,k,bits):
    with pytest.raises(ValueError):
        certify_dtlz6_radial_interval_pf(
            positions,tail,k=k,root_bits=bits)
