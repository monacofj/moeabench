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

from moeabench.diagnostics import auditor, baselines
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
