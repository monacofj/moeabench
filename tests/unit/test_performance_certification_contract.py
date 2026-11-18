# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Binary statistical performance certification, review, and six-case family."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from moeabench import certification as cert


@pytest.fixture
def case():
    return cert._default_cases()[0]


def test_preregistered_family_and_independent_pilot(case):
    assert len(cert.CASES) == 6
    assert len(case["seeds"]) == 64
    assert case["seeds"][:8] == [424242 + 104729*i for i in range(8)]
    assert cert.PILOT_SEEDS == 8
    assert cert.FAMILY_ALPHA == 0.01
    assert cert.TARGET_POWER == 0.95
    assert cert.MARGIN_FRACTION == 0.001


def test_equal_performance_certified_and_material_regression_blocked(case):
    baseline = [0.4]*64
    approved = cert.assess_paired(case, [0.4]*8, [0.4]*8, baseline)
    assert approved["decision"] == "CERTIFIED"
    assert approved["ci_upper"] < approved["margin"]
    assert approved["confirmatory_runs"] == 8
    rejected = cert.assess_paired(case, [0.4]*8, [0.41]*8, baseline)
    assert rejected["decision"] == "NOT_CERTIFIED"
    assert rejected["reason"] == "REGRESSION_DETECTED"
    assert rejected["ci_lower"] > rejected["margin"]


def test_inconclusive_does_not_certify(case):
    baseline = [0.4]*64
    # High variability around the margin, limited to eight
    # confirmatory observations because the independent pilot is deterministic.
    offsets = [0.0, 0.0008]*4
    result = cert.assess_paired(case, [0.4]*8,
                                [0.4 + value for value in offsets], baseline)
    assert result["decision"] == "NOT_CERTIFIED"
    assert result["reason"] == "INSUFFICIENT_EVIDENCE"
    assert result["ci_lower"] < result["margin"] < result["ci_upper"]


def test_high_pilot_variability_blocks_without_running_all_seeds(case):
    baseline = [0.4]*64
    report = cert.assess_paired(case, [0.05, 0.8]*4, [], baseline)
    assert report["decision"] == "NOT_CERTIFIED"
    assert report["reason"] == "INSUFFICIENT_POWER"


def test_simultaneous_inference_is_stricter_than_single_case(case):
    n = 8
    multiple = stats.t.ppf(1 - cert.FAMILY_ALPHA/(2*len(cert.CASES)), n-1)
    single = stats.t.ppf(1 - cert.FAMILY_ALPHA/2, n-1)
    assert multiple > single
    assert cert._required_confirmatory(0.0, 0.0004, 6) == 8
    assert cert._required_confirmatory(0.4, 0.0004, 6) is None


def test_input_safety_rejects_missing_mismatched_negative_or_nonfinite(case):
    baseline = [0.4]*64
    for samples in ([0.4]*7, [np.nan]*8, [-0.1]*8):
        with pytest.raises(ValueError):
            cert.assess_paired(case, samples, [0.4]*8, baseline)
    for samples in ([0.4]*7, [np.inf]*8, [-0.1]*8):
        with pytest.raises(ValueError):
            cert.assess_paired(case, [0.4]*8, samples, baseline)


def test_frozen_reference_requires_real_external_approval(tmp_path):
    src = Path(__file__).resolve().parents[2] / "moeabench/resources/certifications/0.18.0/performance-reference.json"
    doc = json.loads(src.read_text())
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="approval"):
        cert.load_performance_reference(path)
    doc["status"] = "APPROVED"
    doc["approval"] = {"reviewed_by": "independent-code-review",
                       "evidence_url": "https://github.com/monacofj/moeabench/actions/runs/44401850675"}
    path.write_text(json.dumps(doc))
    assert cert.load_performance_reference(path)["status"] == "APPROVED"
    doc["cases"][0]["seeds"][0] = 42
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="seeds|configuration"):
        cert.load_performance_reference(path)


def test_frozen_reference_rejects_checksum_mismatch(tmp_path):
    src = Path(__file__).resolve().parents[2] / "moeabench/resources/certifications/0.18.0/performance-reference.json"
    doc = json.loads(src.read_text())
    doc["reference_fronts"]["DTLZ2"]["points"][0][0] += 0.1
    p = tmp_path / "changed.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="checksum"):
        cert.load_performance_reference(p, require_approved=False)


def test_candidate_cannot_be_certified_against_its_own_release(monkeypatch, tmp_path):
    import moeabench.system as system
    monkeypatch.setattr(cert, "load_performance_reference", lambda path: {
        "source_version": "0.18.0"
    })
    monkeypatch.setattr(system, "version", lambda show=False: "0.18.0")
    with pytest.raises(ValueError, match="distinct"):
        cert.certify_performance("historical.json", tmp_path/"certificate.json")
