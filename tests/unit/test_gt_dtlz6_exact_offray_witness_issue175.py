# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive DTLZ6 offray strict interval certificate and bounded search tests."""
import pytest

from scripts.audit.dtlz6_exact_offray_witness import (
    certify_dtlz6_offray_dominator,replay_dtlz6_offray_dominator,
)
from scripts.audit.dtlz6_selected_offray_search import search_selected_offray
from scripts.audit.dtlz56_exact_same_ray import _decode


def test_exact_irrational_root_offray_strict_dominance():
    original=(.25,.5,.5,1.)
    witness=(.25,.5,.5,0.)
    p=certify_dtlz6_offray_dominator(original,witness)
    assert p["conclusion"]=="rigorously_strictly_dominated"
    assert replay_dtlz6_offray_dominator(original,witness,p)
    assert all(_decode(v)>0
               for v in p["strict_original_objective_separation_lower"])
    assert p["scientific_certification"]=="unverified"


def test_equal_source_and_forged_separation_abstain():
    original=(.25,.5,.5,.25)
    p=certify_dtlz6_offray_dominator(original,original)
    assert p["conclusion"]=="unverified"
    assert not replay_dtlz6_offray_dominator(original,original,p)
    valid=certify_dtlz6_offray_dominator(
        (0.,.5,.5,1.),(0.,.5,.5,0.))
    assert valid["conclusion"]=="unverified"  # last objective is equal zero


def test_numeric_search_never_promotes_unsupported_status():
    ps=[(.25,.5,.5,1.),(.5,1.,0.,.25)]
    r=search_selected_offray(ps,indices=(0,),maxiter=20)
    assert r["target_indices"]==[0]
    assert r["scientific_certification"]=="unverified"
    assert set(r["still_unverified_indices"]).isdisjoint(
        r["rigorously_dominated_indices"])
    assert len(r["rows"])==1
    row=r["rows"][0]
    if row["conclusion"]=="rigorously_strictly_dominated":
        witness=tuple(float.fromhex(v) for v in row["certified_witness_hex"])
        assert replay_dtlz6_offray_dominator(
            ps[0],witness,row["exact_rational_proof"])


@pytest.mark.parametrize("source,witness",[
    ((),(.1,.5,.5,0.)),
    ((.1,.5,.5,0.),()),
    ((.1,.5,.5,float("nan")),(.1,.5,.5,0.)),
    ((.1,.5,.5,1.1),(.1,.5,.5,0.)),
])
def test_invalid_candidates_fail_closed(source,witness):
    with pytest.raises(ValueError):
        certify_dtlz6_offray_dominator(source,witness)


def test_search_wrong_source_fails_closed():
    with pytest.raises(ValueError):
        search_selected_offray([])
    with pytest.raises(ValueError):
        search_selected_offray([(.25,.5,.5,1.)],indices=(6,))
