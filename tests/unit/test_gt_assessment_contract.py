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


@pytest.mark.parametrize("klass,m", [
    (mb.mops.DTLZ7, 2), (mb.mops.DTLZ7, 3),
    (mb.mops.DTLZ8, 2), (mb.mops.DTLZ8, 3),
    (mb.mops.DTLZ8, 4), (mb.mops.DTLZ9, 2),
    (mb.mops.DTLZ9, 3), (mb.mops.DPF1, 3),
])
def test_more_published_geometry_oracles_and_probes(klass, m):
    mop = klass(M=m)
    result = mop.assess_gt(n_points=48, probes=48, seed=29)
    assert result.validity.status == "pass"
    assert result.coverage.status == "unverified"
    assert np.isfinite(result.coverage.value["max_probe_gap"])
    assert result.components.status in ("pass", "unverified")


@pytest.mark.parametrize("klass,m", [
    (mb.mops.DTLZ7, 3), (mb.mops.DTLZ8, 3),
    (mb.mops.DTLZ9, 3), (mb.mops.DPF1, 3),
])
def test_independent_analytic_oracles_detect_corrupt_reference(klass, m):
    mop = klass(M=m)
    front = mop.pf(n_points=48)
    front[-1] += 0.17
    result = mop.assess_gt(reference=front, probes=16)
    assert result.validity.status == "fail"


def test_dtlz8_missing_patch_is_visible_and_fails_occupancy():
    mop = mb.mops.DTLZ8(M=3)
    front = mop.pf(n_points=48)
    front = front[front[:, -1] >= 1/3]
    result = mop.assess_gt(reference=front, probes=16)
    assert result.validity.status == "pass"
    assert result.components.status == "fail"


def test_dtlz7_budget_below_number_of_regions_is_unverified_not_failed():
    mop = mb.mops.DTLZ7(M=7)
    result = mop.assess_gt(n_points=24, probes=16)
    assert result.validity.status == "pass"
    assert result.components.status == "unverified"
    assert result.components.value["expected"] == 64
    assert result.components.value["represented"] == 24


@pytest.mark.parametrize("name,dimensions", [
    ("DPF1", {"M": 3, "D": 2}),
    ("DPF2", {"M": 3, "D": 2}),
    ("DPF2", {"M": 5, "D": 3}),
    ("DPF3", {"M": 3, "D": 2}),
    ("DPF3", {"M": 5, "D": 3}),
    ("DPF4", {"M": 3, "D": 2}),
    ("DPF4", {"M": 5, "D": 3}),
    ("DPF5", {"M": 3, "D": 2}),
    ("DPF5", {"M": 5, "D": 3}),
])
def test_independent_dpf_published_geometry_oracles(name, dimensions):
    mop = getattr(mb.mops, name)(**dimensions)
    result = mop.assess_gt(n_points=64, probes=32, seed=31)
    assert result.validity.status == "pass"
    assert result.coverage.status == "unverified"
    assert np.isfinite(result.coverage.value["max_probe_gap"])
    assert result.cardinality.status == "pass"


@pytest.mark.parametrize("name,dimensions", [
    ("DPF2", {"M": 3, "D": 2}),
    ("DPF3", {"M": 3, "D": 2}),
    ("DPF4", {"M": 3, "D": 2}),
    ("DPF5", {"M": 3, "D": 2}),
])
def test_independent_dpf_oracles_reject_off_front_data(name, dimensions):
    mop = getattr(mb.mops, name)(**dimensions)
    front = mop.pf(n_points=64)
    front[-1] = front[-1] + np.array([0.125, 0.3125, 0.5])
    assert mop.assess_gt(reference=front, probes=16).validity.status == "fail"


def test_dpf5_missing_p_branch_fails_component_check():
    mop = mb.mops.DPF5(M=3, D=2)
    front = mop.pf(n_points=64)
    d_only = front[front[:, -1] >= .5]
    outcome = mop.assess_gt(reference=d_only, probes=16)
    assert outcome.validity.status == "pass"
    assert outcome.components.status == "fail"
    assert outcome.components.value["P"] == 0


def test_nonfinite_reference_is_reported_as_invalid_not_certified():
    mop = mb.mops.DTLZ2(M=2)
    front = mop.pf(n_points=16)
    front[0, 0] = np.nan
    result = mop.assess_gt(reference=front, probes=16)
    assert result.validity.status == "fail"
    assert result.validity.value["nonfinite_coordinates"] == 1
    assert result.coverage.status == "unverified"


def test_diagnostics_are_read_only_and_serializable():
    import json
    from dataclasses import FrozenInstanceError
    mop = mb.mops.DTLZ2(M=2)
    result = mop.assess_gt(n_points=16, probes=16)
    with pytest.raises(TypeError):
        result.coverage.value["max_probe_gap"] = 0
    with pytest.raises(FrozenInstanceError):
        result.count = 0
    as_data = result.to_dict()
    json.dumps(as_data)
    assert as_data["coverage"]["value"]["max_probe_gap"] >= 0


@pytest.mark.parametrize("name", ["DTLZ5", "DTLZ6"])
def test_unproven_full_pareto_geometry_is_not_falsely_certified(name):
    mop = getattr(mb.mops, name)(M=4)
    result = mop.assess_gt(n_points=2, probes=16)
    assert result.validity.status == "unverified"
    assert result.coverage.status == "unverified"


@pytest.mark.parametrize("klass,configuration", [
    (mb.mops.DTLZ7, {"M": 3}),
    (mb.mops.DTLZ8, {"M": 3}),
    (mb.mops.DPF2, {"M": 5, "D": 3}),
    (mb.mops.DPF5, {"M": 3, "D": 2}),
])
def test_componentwise_quality_preserves_region_identity(klass, configuration):
    mop = klass(**configuration)
    result = mop.assess_gt(n_points=64, probes=64, seed=23)
    by_region = result.coverage.value["by_component"]
    spacing = result.regularity.value["by_component"]
    assert len(by_region) > 1
    assert len(spacing) > 1
    assert all(v["probes"] > 0 for v in by_region.values())
    assert all(v["points"] > 0 for v in by_region.values())
    assert all(np.isfinite(v["max_probe_gap"]) for v in by_region.values())
    assert all(v["mean_nearest_neighbor"] is not None for v in spacing.values()
               if v["count"] > 1)
    import json
    json.dumps(result.to_dict())
    with pytest.raises(TypeError):
        next(iter(by_region.values()))["max_probe_gap"] = 0


def test_missing_component_has_no_fabricated_probe_gap():
    mop = mb.mops.DTLZ8(M=3)
    front = mop.pf(n_points=48)
    only_line = front[front[:, -1] >= 1/3]
    result = mop.assess_gt(reference=only_line, probes=64, seed=23)
    assert result.components.status == "fail"
    assert result.coverage.value["by_component"]["patch_0"]["points"] == 0
    assert result.coverage.value["by_component"]["patch_0"]["max_probe_gap"] is None
