# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent scientific-property checks for the opt-in GT assessment API."""
import numpy as np
import pytest
import moeabench as mb


@pytest.mark.parametrize("klass", [mb.mops.DTLZ1, mb.mops.DTLZ2,
                                   mb.mops.DTLZ3, mb.mops.DTLZ4])
@pytest.mark.parametrize("m", [2, 3, 5])
def test_original_analytical_gt_membership_and_observed_quality(klass, m):
    mop = klass(M=m)
    result = mop.assess_gt(n_points=64, probes=48, seed=17)
    assert result.provenance == "analytical_candidate"
    assert result.cardinality.status == "pass"
    assert result.validity.status == "pass"
    assert result.components.status == "pass"
    assert result.coverage.status == "unverified"
    assert result.coverage.value["max_probe_gap"] >= 0
    assert result.regularity.status == "unverified"
    assert result.regularity.value["mean_nearest_neighbor"] > 0
    assert result.feasibility.status == "unverified"
    assert result.replay.status == "unverified"
    assert result.to_dict()["validity"]["basis"].startswith("independent")
    assert not hasattr(mop, "_gt_reference")


def test_independent_oracle_catches_true_off_front_candidate():
    mop = mb.mops.DTLZ2(M=2)
    front = mop.pf(n_points=32)
    front[0] = [0.8, 0.8]
    result = mop.assess_gt(reference=front, probes=16)
    assert result.validity.status == "fail"
    assert result.validity.value["violations"] >= 1
    assert result.components.status == "fail"


def test_simplex_oracle_rejects_wrong_total_even_with_positive_coordinates():
    mop = mb.mops.DTLZ1(M=3)
    front = mop.pf(n_points=32)
    front[-1] *= 1.2
    result = mop.assess_gt(reference=front, probes=16)
    assert result.validity.status == "fail"


def test_exact_duplicate_is_measured_and_detected():
    mop = mb.mops.DTLZ2(M=2)
    front = mop.pf(n_points=16)
    front[-1] = front[0]
    result = mop.assess_gt(reference=front, probes=16)
    assert result.uniqueness.status == "fail"
    assert result.unique_count == 15
    assert result.uniqueness.value["duplicates"] == 1


def test_missing_known_extreme_is_detected_without_fake_coverage_failure():
    mop = mb.mops.DTLZ2(M=3)
    front = mop.pf(n_points=32)[3:]
    result = mop.assess_gt(reference=front, probes=16)
    assert result.validity.status == "pass"
    assert result.components.status == "fail"
    assert result.coverage.status == "unverified"


def test_custom_problem_does_not_inherit_certified_spherical_oracle():
    class ModifiedDTLZ2(mb.mops.DTLZ2):
        pass
    mop = ModifiedDTLZ2(M=3)
    result = mop.assess_gt(n_points=32, probes=16)
    assert result.validity.status == "unverified"
    assert result.components.status == "unverified"
    assert result.coverage.status == "unverified"


def test_external_gt_and_original_reference_are_not_mutated():
    mop = mb.mops.DTLZ2(M=2)
    front = mop.pf(n_points=24)
    before = front.copy()
    first = mop.assess_gt(reference=front, probes=24, seed=9)
    second = mop.assess_gt(reference=front, probes=24, seed=9)
    np.testing.assert_array_equal(front, before)
    assert first.to_dict() == second.to_dict()


def test_calibrated_gt_report_contains_assessment_without_side_effects(tmp_path):
    mop = mb.mops.DTLZ2(M=2)
    path = tmp_path / "gt.json"
    assert mop.calibrate(source_baseline=path, n_points=32)
    original_gt = mop.calibrated_gt
    before = path.read_bytes()
    md = mop.report(show=False, markdown=True, probes=24)
    plain = mop.report(show=False, markdown=False, probes=24)
    assert "GT assessment" in md and "GT assessment" in plain
    assert "validity" in plain and "coverage" in plain
    assert "unverified" in plain
    assert "text" in md
    np.testing.assert_array_equal(mop.calibrated_gt, original_gt)
    assert path.read_bytes() == before


def test_uncalibrated_report_never_creates_gt_or_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mop = mb.mops.DTLZ2(M=2)
    message = mop.report(show=False, markdown=False)
    assert "Not calibrated" in message
    assert not hasattr(mop, "_gt_reference")
    assert not list(tmp_path.iterdir())


def test_report_can_skip_assessment(tmp_path, monkeypatch):
    mop = mb.mops.DTLZ1(M=2)
    mop.calibrate(source_baseline=tmp_path / "gt.json", n_points=16)
    monkeypatch.setattr(mop, "assess_gt", lambda **kwargs:
                        pytest.fail("assessment must not be invoked"))
    report = mop.report(show=False, assess=False)
    assert "assessment not requested" in report


@pytest.mark.parametrize("kwargs", [
    {"probes": 1}, {"seed": -1}, {"tolerance": 0},
    {"tolerance": np.nan},
])
def test_invalid_protocol_inputs_are_rejected(kwargs):
    mop = mb.mops.DTLZ2(M=2)
    with pytest.raises(ValueError):
        mop.assess_gt(n_points=32, **kwargs)


def test_reference_and_analytical_budget_are_mutually_exclusive():
    mop = mb.mops.DTLZ2(M=2)
    with pytest.raises(ValueError):
        mop.assess_gt(reference=np.ones((3, 2)), n_points=32)


def test_all_built_in_mops_offer_generic_assessment_without_fake_proof():
    names = ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5",
             "DTLZ6", "DTLZ7", "DTLZ8", "DTLZ9", "DPF1",
             "DPF2", "DPF3", "DPF4", "DPF5")
    for name in names:
        mop = getattr(mb.mops, name)()
        outcome = mop.assess_gt(n_points=32, probes=16)
        assert outcome.count > 0
        assert outcome.cardinality.status in ("pass", "fail")
        assert outcome.validity.status in ("pass", "fail", "unverified")
        assert outcome.coverage.status == "unverified"
