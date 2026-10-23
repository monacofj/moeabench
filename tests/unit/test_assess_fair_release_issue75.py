# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #75: fail-closed compatibility release gate cannot retag provenance."""
import copy
import pytest

from calibration.scripts.assess_fair_release_issue75 import assess_release
from test_fair_release_preview_issue75 import _manifest, _q, _history


def _env(monkeypatch, version_status="MISMATCH", stability_status="PASSED"):
    monkeypatch.setattr("moeabench.system.version", lambda show=False: "0.16.1")
    return {
        "library_version":"0.16.1",
        "version_check":{"status":version_status,
                         "differences":{"numpy":{"certified":"old","current":"new"}}},
        "stability_check":{"status":stability_status,
                           "checks":{"python_random":"PASS",
                                     "numpy_generator":"PASS"}},
        "runtime":{"python":"3.14.4","numpy":"2.5.4"},
    }


def test_separate_source_baseline_version_from_library_version(monkeypatch):
    env=_env(monkeypatch)
    m=_manifest()
    m["source_version"]="0.14.0"
    result=assess_release(m,_q(),_history(),env)
    assert result["status"]=="BLOCKED_PENDING_SCIENTIFIC_RELEASE_REVIEW"
    assert result["certified"] is False
    assert result["publishable"] is False
    assert result["overwrite_canonical_allowed"] is False
    assert result["calibration_provenance"]["original_baseline_version"]=="0.14.0"
    assert result["calibration_provenance"]["consumer_library_version"]=="0.16.1"
    assert result["checks"]["numerical_stability_smoke_passed"] is True
    assert result["paired_q_impact"]["allowable_drift_threshold"] is None
    blockers={x["code"] for x in result["release_blockers"]}
    assert "ARCHIVED_BASELINE_LIBRARY_VERSION_DIFFER" in blockers
    assert "DEPENDENCY_CERTIFICATION_MISSING" in blockers
    assert "BREAKING_Q_CHANGE_REQUIRES_REVIEW" in blockers
    assert "HISTORICAL_P0_UNOBSERVABLE" in blockers


def test_no_release_without_stability_even_when_versions_match(monkeypatch):
    env=_env(monkeypatch,version_status="MATCH",stability_status="FAILED")
    m=_manifest()
    m["source_version"]="0.16.1"
    result=assess_release(m,_q(),_history(),env)
    assert result["checks"]["source_calibration_and_runtime_version_equal"] is True
    assert result["checks"]["numerical_stability_smoke_passed"] is False
    assert "NUMERICAL_STABILITY_UNVERIFIED_OR_FAILED" in {
        x["code"] for x in result["release_blockers"]
    }
    assert result["publishable"] is False


def test_fail_closed_on_wrong_hash_measure_history_or_environment(monkeypatch):
    env=_env(monkeypatch)
    m=_manifest()
    m["source_version"]="0.14.0"
    q=_q()
    q["baseline_candidate_sha256"]="spoofed"
    with pytest.raises(ValueError,match="hashes"):
        assess_release(m,q,_history(),env)
    q=_q()
    q["informative_checkpoint_count"]=0
    with pytest.raises(ValueError,match="nonclipped"):
        assess_release(m,q,_history(),env)
    h=_history()
    h["report_generation_safe"]=True
    with pytest.raises(ValueError,match="Historical provenance"):
        assess_release(m,_q(),h,env)
    bad=copy.deepcopy(env)
    bad["stability_check"]["status"]="UNCHECKED"
    with pytest.raises(ValueError,match="actually measured"):
        assess_release(m,_q(),_history(),bad)
    bad=copy.deepcopy(env)
    bad["library_version"]="0.14.0"
    with pytest.raises(ValueError,match="not for this MoeaBench"):
        assess_release(m,_q(),_history(),bad)


def test_reject_mop_reclassification_without_scientific_provenance(monkeypatch):
    env=_env(monkeypatch)
    m=_manifest()
    m["source_version"]="0.14.0"
    m["problems"]["DPF5"]["status"]="PROVISIONAL_NOT_CERTIFIED"
    with pytest.raises(ValueError,match="scope"):
        assess_release(m,_q(),_history(),env)
