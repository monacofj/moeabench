# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Issue #75: independent probes of FAIR's GT and resolution contracts.

These tests intentionally separate a mathematical oracle from the currently
broken production paths. The two original RED reproductions now check their fixed contracts
directly; a regression must once again fail rather than be accepted as XFAIL.
The tests are small and never run an MOEA or regenerate baseline artifacts.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from moeabench.diagnostics import auditor, baselines, fair, calibration, qscore
from moeabench.diagnostics.enums import DiagnosticStatus
from moeabench.diagnostics.utils import _resolve_diagnostic_context


def _line(n=41):
    """A small, explicitly known two-objective reference line."""
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack((x, 1.0 - x))


def _fps_reference_oracle(gt, k):
    """Plain independent farthest-point sampling with seed-zero first point.

    The resolver test uses no production s_K implementation as its oracle.
    The start and tie-breaking mirror the specified deterministic seed-zero FPS.
    """
    rng = np.random.RandomState(0)
    selected = [int(rng.randint(0, len(gt)))]
    closest = np.linalg.norm(gt - gt[selected[0]], axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(closest))
        selected.append(idx)
        closest = np.minimum(closest, np.linalg.norm(gt - gt[idx], axis=1))
    return gt[selected]


def _spacing_oracle(points):
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    return float(np.median(np.min(d, axis=1)))


def test_independent_resolution_oracle_has_non_unit_scale():
    """Positive check: ten FPS points on a unit line do not have s_K == 1."""
    gt = _line()
    expected = _spacing_oracle(_fps_reference_oracle(gt, 10))
    assert 0.0 < expected < 1.0


def test_resolver_computes_documented_resolution_from_explicit_gt():
    """No MOP override: s_K must come from FPS on the selected GT at K."""
    gt = _line()
    population = gt[::4]  # eleven non-dominated points
    k = len(population)
    expected = _spacing_oracle(_fps_reference_oracle(gt, k))
    actual = _resolve_diagnostic_context(population, ref=gt)["s_k"]
    assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_audit_preserves_explicit_gt_when_registry_has_another_reference(monkeypatch):
    """Stop the real audit at U_K construction and inspect its actual input GT."""
    explicit_gt = _line(21)
    registered_gt = explicit_gt + np.array([0.0, 4.0])
    population = explicit_gt[::3]
    captured = {}

    def offline_baselines(**kwargs):
        return {"_gt_registry": {"Unknown__M2": registered_gt}, "problems": {}}

    class StopBeforeMetrics(Exception):
        pass

    def capture_reference(gt, k, seed=0):
        captured["gt"] = np.asarray(gt).copy()
        raise StopBeforeMetrics

    monkeypatch.setattr(baselines, "load_offline_baselines", offline_baselines)
    monkeypatch.setattr(baselines, "get_ref_uk", capture_reference)
    with pytest.raises(StopBeforeMetrics):
        auditor.audit(population, ground_truth=explicit_gt, quality=False)

    np.testing.assert_array_equal(captured["gt"], explicit_gt)


def test_fps_subset_cannot_estimate_off_sample_discretization_residue():
    """Demonstrate why current ideal_res calibration necessarily gives zero.

    For any FPS subset of the same GT, each selected point is identical to
    some GT point. By contrast, fresh midpoints of this known analytical line
    have strictly positive nearest-GT distances.
    """
    gt = _line(41)
    subset = baselines.get_ref_uk(gt, 10, seed=999)
    subset_dist = cdist(subset, gt).min(axis=1)
    assert np.allclose(subset_dist, 0.0, atol=1e-14)

    analytic_midpoints = (gt[:-1] + gt[1:]) / 2.0
    independent_dist = cdist(analytic_midpoints, gt).min(axis=1)
    assert np.median(independent_dist) > 0.0


def test_explicit_resolution_override_remains_authoritative():
    """A user-supplied numerical scale is an intentional context override."""
    gt = _line(41)
    population = gt[::4]
    resolved = _resolve_diagnostic_context(population, ref=gt, s_k=0.375)
    assert resolved["s_k"] == pytest.approx(0.375)


def test_audit_explicit_gt_and_scale_are_visible_in_result_context():
    """No hidden replacement or rescaling is allowed in FAIR-only mode."""
    gt = _line(41)
    population = gt[::4]
    result = auditor.audit(
        population, ground_truth=gt, s_k=0.375, quality=False,
    )
    context = result.diagnostic_context
    np.testing.assert_array_equal(context["GT"], gt)
    assert context["gt_source"] == "explicit"
    assert context["s_k"] == pytest.approx(0.375)
    assert context["k"] == len(population)


def test_q_scores_fail_closed_for_explicit_gt_with_incompatible_registry(monkeypatch):
    """An otherwise valid physical audit must not borrow a different GT's Q baselines."""
    explicit_gt = _line(41)
    registered_gt = explicit_gt * 2.0
    population = explicit_gt[::4]

    def offline_baselines(**kwargs):
        return {
            "_gt_registry": {"Unknown__M2": registered_gt},
            "problems": {},
            "version": "test",
        }

    monkeypatch.setattr(baselines, "load_offline_baselines", offline_baselines)

    result = auditor.audit(population, ground_truth=explicit_gt)
    assert result.status == DiagnosticStatus.MISSING_BASELINE
    assert result.quality is None
    assert result.fr is not None
    assert set(result.fr.metrics) == {
        "CLOSENESS", "COVERAGE", "GAP", "REGULARITY", "BALANCE", "HEADWAY"
    }
    assert result.diagnostic_context["baseline_gt_compatible"] is False

    physical_only = auditor.audit(population, ground_truth=explicit_gt, quality=False)
    assert physical_only.fr is not None
    assert physical_only.quality is None


def test_registered_calibration_transform_applies_to_final_and_initial(monkeypatch):
    """The same ideal/nadir transform must be applied to both generations."""
    from types import SimpleNamespace
    from moeabench.diagnostics import utils

    gt = _line(41)
    ideal = np.array([1.0, 2.0])
    nadir = np.array([3.0, 6.0])
    span = nadir - ideal
    final_normalized = gt[::4]
    initial_normalized = final_normalized + 0.3

    class ToyExperiment:
        def __init__(self):
            self.mop = SimpleNamespace(name="Toy", pf=lambda: gt * span + ideal)

        def front(self, generation):
            return (final_normalized if generation == -1 else initial_normalized) * span + ideal

        def pop(self, generation):
            return SimpleNamespace(
                objectives=(final_normalized if generation == -1 else initial_normalized) * span + ideal
            )

    def offline_baselines(**kwargs):
        return {"_gt_registry": {"Toy__M2": gt}, "problems": {}}

    monkeypatch.setattr(baselines, "load_offline_baselines", offline_baselines)
    monkeypatch.setattr(
        utils, "_calibration_coordinates",
        lambda selected_gt, problem: (ideal, nadir)
        if problem == "Toy" and np.array_equal(selected_gt, gt) else None,
    )
    resolved = _resolve_diagnostic_context(ToyExperiment())
    np.testing.assert_allclose(resolved["P_final"], final_normalized)
    np.testing.assert_allclose(resolved["P_initial"], initial_normalized)
    np.testing.assert_array_equal(resolved["GT"], gt)
    assert resolved["normalization"]["method"] == "calibration_ideal_nadir"


def test_closeness_uses_nonzero_calibrated_residue(monkeypatch):
    """Verify a nonzero ideal_res actually influences both FR scalar and points."""
    gt = _line(41)
    population = gt[::4] + np.array([0.03, 0.03])
    scale, residue = 0.15, 0.20
    monkeypatch.setattr(baselines, "reference_compatibility", lambda problem, gt: True)
    monkeypatch.setattr(
        baselines, "get_baseline_data",
        lambda problem, k, metric: {"ideal_res": residue},
    )
    distances = cdist(population, gt).min(axis=1)
    expected_points = np.maximum(0.0, distances / scale - residue)
    result = fair.closeness(population, ref=gt, s_k=scale, problem="Toy", k=len(population))
    np.testing.assert_allclose(result.raw_data, expected_points, atol=1e-12)
    assert float(result.value) == pytest.approx(float(np.median(expected_points)))


def test_closeness_rejects_invalid_calibrated_residue(monkeypatch):
    """Malformed or negative residue must not be treated as zero silently."""
    gt = _line(41)
    monkeypatch.setattr(baselines, "reference_compatibility", lambda problem, gt: True)
    monkeypatch.setattr(
        baselines, "get_baseline_data",
        lambda problem, k, metric: {"ideal_res": -0.1},
    )
    with pytest.raises(ValueError, match="ideal_res"):
        fair.closeness(gt[::4], ref=gt, s_k=0.15, problem="Toy", k=11)


def test_fresh_analytical_ideal_sample_excludes_deterministic_gt_prefix():
    """DTLZ1.ps() restarts seed=42 each call; its first N points repeat exactly."""
    from moeabench.mops import DTLZ1
    gt = DTLZ1(M=3).pf(n_points=100)
    sample = calibration.independent_analytical_reference(DTLZ1(M=3), gt, n_fresh=40)
    assert sample is not None
    assert sample.shape == (40, 3)
    distances = cdist(sample, gt).min(axis=1)
    assert np.all(distances > 1e-10)


def test_ideal_residue_uses_independent_analytic_points_not_same_gt_subset():
    """A tiny analytical reference has a directly verifiable positive residue."""
    gt = _line(41)
    ideal = (gt[:-1] + gt[1:]) / 2
    s_k = 0.25
    expected = float(np.median(cdist(ideal, gt).min(axis=1))) / s_k
    assert calibration.estimate_ideal_residue(gt, ideal, s_k) == pytest.approx(expected)
    assert expected > 0.0
    with pytest.raises(ValueError, match="overlaps finite GT"):
        calibration.estimate_ideal_residue(gt, gt[::4], s_k)


def test_ideal_residue_unavailable_for_empirical_gt_is_not_disguised_as_estimate(monkeypatch):
    """Non-analytical GT gets an explicit provenance tag with its zero correction."""
    monkeypatch.setattr(calibration, "N_SAMPLES_ECDF", 3)
    monkeypatch.setattr(calibration, "N_IDEAL_DRAWS", 2)
    gt = _line(41)
    baseline = calibration._generate_baselines(
        "Toy", gt, [10], independent_ideal=None
    )["10"]["closeness"]
    assert baseline["ideal_res"] == 0.0
    assert baseline["ideal_res_method"] == "unavailable_no_independent_ideal"
    midpoints = (gt[:-1] + gt[1:]) / 2
    calibrated = calibration._generate_baselines(
        "Toy", gt, [10], independent_ideal=midpoints
    )["10"]["closeness"]
    assert calibrated["ideal_res"] > 0.0
    assert calibrated["ideal_res_method"] == "independent_analytical_pf"


def test_staged_residue_migration_shifts_ecdf_without_modifying_legacy_input():
    """All Q-Closeness baseline samples must receive the physical correction."""
    from calibration.scripts.migrate_closeness_residue import residue_update_record
    old = {
        "ideal_res": 0.0,
        "rand_ecdf": [0.03, 0.20, 0.8],
        "rand50": 0.20,
        "p95": 0.75,
        "sigma": 0.4,
    }
    after = residue_update_record(old, 0.10)
    np.testing.assert_allclose(after["rand_ecdf"], [0.0, 0.10, 0.70])
    assert after["rand50"] == pytest.approx(0.10)
    assert after["p95"] == pytest.approx(0.65)
    assert after["sigma"] == old["sigma"]
    assert old["ideal_res"] == 0.0  # Frozen historical artifact untouched.
    with pytest.raises(ValueError, match="already been corrected"):
        residue_update_record(after, 0.10)


def test_physical_closeness_mismatched_gt_never_borrows_other_residue(monkeypatch):
    """A physical metric with custom GT remains uncorrected unless calibrated."""
    gt = _line(41)
    pop = gt[::4] + np.array([0.03, 0.03])
    scale = 0.15
    monkeypatch.setattr(baselines, "reference_compatibility", lambda problem, ref: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("An incompatible baseline residue was requested")

    monkeypatch.setattr(baselines, "get_baseline_data", forbidden)
    physical = fair.closeness(pop, ref=gt, s_k=scale, problem="Toy", k=len(pop))
    expected = cdist(pop, gt).min(axis=1) / scale
    np.testing.assert_allclose(physical.raw_data, expected)

    overridden = fair.closeness(
        pop, ref=gt, s_k=scale, problem="Toy", k=len(pop), ideal_res=0.10
    )
    np.testing.assert_allclose(
        overridden.raw_data, np.maximum(0.0, expected - 0.10)
    )


def test_pointwise_closeness_visual_scores_use_same_calibrated_residue(monkeypatch):
    """Plot symbols must not use raw distances when FAIR subtracts ideal_res."""
    gt = _line(41)
    pop = gt[::4] + np.array([0.03, 0.03])
    scale, epsilon, bad_median = 0.15, 0.20, 2.0
    monkeypatch.setattr(baselines, "reference_compatibility", lambda problem, ref: True)
    monkeypatch.setattr(
        baselines, "get_baseline_data",
        lambda problem, k, metric: {"ideal_res": epsilon},
    )
    monkeypatch.setattr(baselines, "snap_k", lambda k, problem=None: k)
    monkeypatch.setattr(
        baselines, "get_baseline_values",
        lambda problem, k, metric, target_m=None: (0.0, bad_median),
    )
    physical = fair.closeness(pop, ref=gt, s_k=scale, problem="Toy", k=len(pop))
    expected = np.clip(
        1.0 - np.log1p(physical.raw_data) / np.log1p(bad_median), 0.0, 1.0
    )
    actual = qscore.q_closeness_points(
        pop, ref=gt, s_k=scale, problem="Toy", k=len(pop)
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert np.all((0 <= actual) & (actual <= 1))


def test_q_closeness_responds_to_residue_in_nondegenerate_calibration(monkeypatch):
    """Wasserstein Q must react when identical physical and null scales shift.

    Fixed independent oracle: observed values 1, bad reference values 2
    give W1(observed, ideal=0)=1 and W1(observed, bad)=1; Q=1/2.
    After a finite-GT correction 0.2 applied to both observed and bad
    samples, the distances are 0.8 and 1 respectively, giving Q=1/1.8.
    Unlike an optimizer run clipped at Q=0, this isolates score sensitivity.
    """
    monkeypatch.setattr(baselines, "snap_k", lambda k, problem=None: k)
    reference = {"bad": 2.0}

    def ecdf(problem, k, metric, target_m=None):
        return 0.0, reference["bad"], np.full(200, reference["bad"])

    monkeypatch.setattr(baselines, "get_baseline_ecdf", ecdf)
    original = qscore.q_closeness(np.ones(50), problem="Toy", k=50)
    assert original.value == pytest.approx(0.5)

    reference["bad"] = 1.8
    corrected = qscore.q_closeness(np.full(50, 0.8), problem="Toy", k=50)
    assert corrected.value == pytest.approx(1.0 / 1.8)
    assert corrected.value > original.value
