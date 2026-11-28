# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive original DTLZ6 selected proof-obstruction diagnostic tests."""
import pytest
from scripts.audit.dtlz6_selected_support_obstruction import (
    explain_original_selected,replay_explanation,
)


def test_exact_support_obstruction_uses_historical_source_and_abstains():
    ps=[
        (0.,1.,0.,1/3),
        (.125,1.,0.,.25),
        (.5,.5,.5,0.),
        (.5,.5,.5,.25),
    ]
    r=explain_original_selected(ps)
    assert replay_explanation(ps,r)
    assert r["selected_count"]==4
    assert r["necessary_conditions_not_sufficient"]
    assert r["scientific_certification"]=="unverified"
    assert set(row["selected_index"] for row in r["row_diagnostics"])==set(
        r["original_historical_inconclusive_indices"])
    assert set(r["still_inconclusive_historical_indices"]).issubset(
        set(r["original_historical_inconclusive_indices"]))
    assert all(row["reason"] in {
        "no_available_sufficient_angular_support_pattern",
        "support_pattern_present_but_no_positive_radial_proof",
        "nonpositive_exact_support_margin",
        "positive_margin_requires_certification_reconciliation",
    } for row in r["row_diagnostics"])


def test_forgery_and_different_source_rejected():
    ps=[(.5,.5,.5,.25)]
    r=explain_original_selected(ps)
    forged=dict(r)
    forged["row_diagnostics"]=[]
    assert not replay_explanation(ps,forged)
    assert not replay_explanation([(.125,.5,.5,.25)],r)


@pytest.mark.parametrize("bad",[[],[(0.,0.,0.)],
                                [(0.,0.,0.,float("nan"))]])
def test_invalid_source_fails_closed(bad):
    with pytest.raises(ValueError):
        explain_original_selected(bad)
