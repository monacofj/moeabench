# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New, independent certification tests: never modify historical regressions."""
from __future__ import annotations

import json
import numpy as np
import pytest

from moeabench import certification as cert


def _synth_case():
    case = cert._default_cases()[0]
    case["observations"] = [
        {"seed": seed, "igdplus": 0.5, "gdplus": 0.5, "emd": 0.5, "front_size": 10}
        for seed in case["seeds"]
    ]
    return case


def test_fixed_study_contains_six_cases_sixty_four_seeds():
    cases = cert._default_cases()
    assert len(cases) == 6
    assert all(len(case["seeds"]) == 64 for case in cases)
    assert all(len(set(case["seeds"])) == 64 for case in cases)
    assert all(case["moea_kwargs"] == {"population": 48, "generations": 60} for case in cases)
    assert cert.FAMILY_ALPHA == 0.01
    assert cert.TARGET_POWER == 0.95
    assert cert.MARGIN_FRACTION == 0.001


def test_comparison_requires_evidence_and_excludes_pilot_from_interval():
    case = _synth_case()
    baseline = [0.5] * 64
    passed = cert.assess_paired(case, [0.5] * 8, [0.5] * 8, baseline)
    assert passed["decision"] == "CERTIFIED"
    assert passed["confirmatory_runs"] == 8
    assert passed["mean_difference"] == pytest.approx(0.0)
    worsening = cert.assess_paired(case, [0.5] * 8, [0.51] * 8, baseline)
    assert worsening["decision"] == "NOT_CERTIFIED"
    assert worsening["reason"] == "REGRESSION_DETECTED"
    with pytest.raises(ValueError, match="Confirmatory"):
        cert.assess_paired(case, [0.5] * 8, [0.5] * 7, baseline)


def test_insufficient_power_does_not_fabricate_acceptance():
    case = _synth_case()
    baseline = [0.5] * 64
    pilot = [0.1, 0.9] * 4
    report = cert.assess_paired(case, pilot, [], baseline)
    assert report["decision"] == "NOT_CERTIFIED"
    assert report["reason"] == "INSUFFICIENT_POWER"
    assert report["required_confirmatory"] is None


def test_baseline_artifact_is_single_write_and_requires_external_approval(
        monkeypatch, tmp_path):
    # A fake measurement engine allows exercising the artifact contract without
    # launching the real 384-run study in routine CI.
    monkeypatch.setattr(cert, "_reference_runtime", lambda: {"moeabench": "0.18.0"})
    monkeypatch.setattr(cert, "_revision", lambda: "0d3e66804351e23b93fb483ff25e178e06ef6e12")
    monkeypatch.setattr(cert, "_measure", lambda case, seed, front: {
        "seed": seed, "front_size": 24, "igdplus": 0.5,
        "gdplus": 0.2, "emd": 0.3
    })
    output = tmp_path / "baseline.json"
    written = cert.create_performance_baseline(output)
    assert written == output
    data = json.loads(output.read_text())
    assert data["status"] == "CANDIDATE_REQUIRES_REVIEW"
    assert len(data["cases"]) == 6
    assert all(len(c["observations"]) == 64 for c in data["cases"])
    with pytest.raises(ValueError, match="approval"):
        cert.load_performance_reference(output)
    with pytest.raises(FileExistsError):
        cert.create_performance_baseline(output)
    # Simulate an independent maintainer approval in this synthetic test only.
    data["status"] = "APPROVED"
    data["approval"] = {"reviewed_by": "independent-reviewer-test", "evidence_url": "https://github.com/example/review/1"}
    output.write_text(json.dumps(data))
    assert cert.load_performance_reference(output)["status"] == "APPROVED"
    data["reference_fronts"]["DTLZ2"]["points"][0][0] += 1
    output.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        cert.load_performance_reference(output)


def test_numeric_history_carries_only_hashes_not_copied_fronts(monkeypatch, tmp_path):
    from moeabench import environment_verification as ev
    monkeypatch.setattr(ev, "snapshot", lambda: {"moeabench": "0.18.0"})
    monkeypatch.setattr(
        ev, "certified_manifest", lambda: {"versions": {"moeabench": "0.18.0"}}
    )
    file = cert.register_numeric_history(
        tmp_path, evidence_url="https://github.com/example/run/123",
        counts={"unit": 332, "integration": 16, "stability": 41}
    )
    obj = json.loads(file.read_text())
    assert obj["version"] == "0.18.0"
    assert len(obj["numerical_fixtures_sha256"]) == 3
    assert all(len(v) == 64 for v in obj["numerical_fixtures_sha256"].values())
    assert "front" not in json.dumps(obj).replace("stability_reference_fronts", "")
    with pytest.raises(FileExistsError):
        cert.register_numeric_history(tmp_path)
