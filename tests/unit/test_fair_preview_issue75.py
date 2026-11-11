# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The report preview must display missing clinical evidence, never fabricate it."""

import pytest
from calibration.scripts.render_fair_evidence_preview import render_preview

METRICS = ("CLOSENESS", "COVERAGE", "GAP", "REGULARITY", "BALANCE", "HEADWAY")


def _pilot():
    return {
        "status": "provisional_longitudinal_pilot_not_certified",
        "experiment": {"moea":"NSGA2","M":3,"population":16,
                       "generations":8,"seed":7},
        "provenance": {"gt_source":"registered","baseline_version":"v0.14.0",
                       "gt_sha256":"gtsha","P0_sha256":"p0sha","Pf_sha256":"pfsha"},
        "sample_sizes": {"P0":16,"Pf":10},
        "physical_metrics": {name:0.125 for name in METRICS},
        "clinical_q_scores_frozen_baseline": {"Q_"+k:0.625 for k in METRICS},
        "independent_headway":0.125,
        "headway_validated_from_true_P0":True,
    }


def _archive():
    return {
        "status": "provisional_noncertified",
        "cannot_be_used_as_certified_audit_report": True,
        "runs": [{
            "study":"DTLZ1_NSGA2", "run":"00",
            "metrics":{**{m:0.5 for m in METRICS}, "HEADWAY":None},
            "q_scores":None,
        }],
        "errors":[],
    }


def test_provisional_renderer_distinguishes_fresh_and_historical_evidence():
    body = render_preview(_pilot(), _archive())
    assert "NOT CERTIFIED" in body
    assert "Initial population: 16" in body
    assert "Independently" not in body or "independently" in body
    assert "Unavailable" in body
    assert "no verified generation-zero population" in body
    assert "0.625000" in body
    assert "<script" not in body


def test_renderer_rejects_unsupported_archived_clinical_scores():
    archive = _archive()
    archive["runs"][0]["q_scores"] = {"Q_CLOSENESS":0.95}
    with pytest.raises(ValueError, match="lack certified"):
        render_preview(_pilot(), archive)


def test_renderer_fails_without_proven_genuine_population():
    pilot = _pilot()
    pilot["headway_validated_from_true_P0"] = False
    with pytest.raises(ValueError, match="genuine P0"):
        render_preview(pilot, _archive())


def test_renderer_can_show_verified_staged_baseline_without_certifying_it():
    pilot = _pilot()
    pilot["candidate_closeness_comparison"] = {
        "other_five_FAIR_metrics_and_Q_unchanged":True,
        "ideal_res":0.04,
        "old_closeness":0.5,
        "new_closeness":0.46,
        "old_q_closeness":0.1,
        "new_q_closeness":0.12,
        "candidate_baseline_sha256":"candidatehash",
    }
    body = render_preview(pilot, _archive())
    assert "Staged calibration comparison" in body
    assert "candidatehash" in body
    assert "0.460000" in body
