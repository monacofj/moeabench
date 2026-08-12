# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect
import importlib

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from moeabench.metrics import evaluator
from moeabench.metrics.GEN_gd import GEN_gd
from moeabench.metrics.GEN_igd import GEN_igd
from moeabench.stats.attainment import AttainmentSurface


REFERENCE = np.array([[0.0, 1.0], [1.0, 0.0], [0.4, 0.4]])
GOOD = np.array([[0.2, 0.8], [0.8, 0.2], [0.45, 0.45]])


def _hv(front, **kwargs):
    return evaluator.hypervolume(front, mode="exact", progress=False, **kwargs)


def test_hypervolume_signature_only_exposes_reference_context_api():
    removed_parameter = "".join(("jo", "int"))
    assert removed_parameter not in inspect.signature(evaluator.hypervolume).parameters


def test_hypervolume_without_ref_is_self_referenced():
    shifted = GOOD + 4.0

    assert float(_hv(GOOD)) == pytest.approx(float(_hv(shifted)))
    assert _hv(GOOD).reference_context == "self"


def test_hypervolume_ref_exclusively_defines_fixed_bounds():
    poor = GOOD * 1.5

    good_hv = float(_hv(GOOD, ref=REFERENCE))
    poor_hv = float(_hv(poor, ref=REFERENCE))

    assert good_hv != pytest.approx(poor_hv)
    assert _hv(GOOD, ref=REFERENCE).reference_context == "external"


def test_radially_scaled_fronts_do_not_collapse_with_same_reference():
    poor = GOOD * 1.5

    assert float(_hv(GOOD)) == pytest.approx(float(_hv(poor)))
    assert float(_hv(GOOD, ref=REFERENCE)) > float(_hv(poor, ref=REFERENCE))


def test_multiple_runs_share_one_self_reference_box():
    run_a = GOOD
    run_b = GOOD * 1.5

    self_values = _hv([run_a, run_b]).values
    with pytest.warns(UserWarning, match="reference bbox is expanded"):
        pooled_values = _hv([run_a, run_b], ref=[run_a, run_b]).values

    assert np.allclose(self_values, pooled_values)
    assert self_values[0, 0] != pytest.approx(self_values[0, 1])


def test_all_scales_reuse_external_reference_bounds(monkeypatch):
    calls = []

    class RecordingHypervolume:
        def __init__(self, fronts, objectives, ideal, nadir, **kwargs):
            calls.append((ideal.copy(), nadir.copy()))
            self.fronts = fronts

        def evaluate(self):
            return np.ones(len(self.fronts))

    class Mop:
        name = "TEST_MOP"

    class Front(np.ndarray):
        pass

    exp = GOOD.view(Front)
    exp.mop = Mop()
    monkeypatch.setattr(evaluator, "GEN_hypervolume", RecordingHypervolume)
    baselines = importlib.import_module("moeabench.diagnostics.baselines")
    monkeypatch.setattr(
        baselines,
        "load_offline_baselines",
        lambda: {"_gt_registry": {"TEST_MOP__M2": REFERENCE.tolist()}},
    )

    for scale in ("raw", "rel", "abs"):
        calls.clear()
        evaluator.hypervolume(exp, ref=REFERENCE, mode="exact", scale=scale, progress=False)
        assert calls
        for ideal, nadir in calls:
            assert np.array_equal(ideal, np.min(REFERENCE, axis=0))
            assert np.array_equal(nadir, np.max(REFERENCE, axis=0))


def test_out_of_reference_points_do_not_expand_bounds():
    beyond_nadir = np.array([[1.2, 1.2]])
    better_than_ideal = np.array([[-0.2, 0.5]])

    with pytest.warns(UserWarning, match="Hypervolume floor saturation"):
        beyond_result = _hv(beyond_nadir, ref=REFERENCE)
    assert float(beyond_result) == 0.0
    assert float(_hv(better_than_ideal, ref=REFERENCE)) > 1.1 * 0.6


def test_degenerate_reference_bounds_are_rejected():
    with pytest.raises(ValueError, match="non-zero range"):
        _hv(GOOD, ref=np.array([[1.0, 1.0]]))


def test_public_metrics_use_ref_as_external_reference():
    expected_emd = np.mean([
        wasserstein_distance(GOOD[:, i], REFERENCE[:, i])
        for i in range(GOOD.shape[1])
    ])

    assert float(evaluator.gd(GOOD, ref=REFERENCE, progress=False)) == pytest.approx(
        GEN_gd([GOOD], REFERENCE).evaluate()[0]
    )
    assert float(evaluator.igd(GOOD, ref=REFERENCE, progress=False)) == pytest.approx(
        GEN_igd([GOOD], REFERENCE).evaluate()[0]
    )
    assert float(evaluator.emd(GOOD, ref=REFERENCE, progress=False)) == pytest.approx(expected_emd)
    assert float(_hv(GOOD, ref=REFERENCE)) == pytest.approx(0.5725)


def test_hypervolume_report_identifies_reference_context():
    self_report = _hv(GOOD).report(show=False, markdown=True)
    fixed_report = _hv(GOOD, ref=REFERENCE).report(show=False, markdown=True)

    assert "Self-Referenced Hypervolume" in self_report
    assert "not directly comparable" in self_report
    assert "Fixed Reference Context" in fixed_report
    assert "same reference" in fixed_report


def test_attainment_ref_point_remains_a_geometric_endpoint():
    surface = AttainmentSurface([[0.2, 0.8], [0.8, 0.2]])

    assert surface.volume(ref_point=[1.0, 1.0]) == pytest.approx(0.28)


def test_external_diagnostics_do_not_change_known_hypervolume():
    result = _hv(GOOD, ref=REFERENCE)

    assert float(result) == pytest.approx(0.5725)
    assert result.diagnostics["nominal_bbox_volume"] == pytest.approx(1.1 ** 2)
    assert result.diagnostics["raw_hv_fraction_of_nominal_bbox"][0] == pytest.approx(
        0.5725 / (1.1 ** 2)
    )


def test_dominated_reference_points_without_bound_expansion_do_not_warn():
    ref = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.5],
        [0.8, 0.8],
    ])

    with warnings_not_matching("reference bbox is expanded"):
        result = _hv(GOOD, ref=ref)

    assert result.diagnostics["reference_dominated_fraction"] == pytest.approx(0.25)
    assert result.diagnostics["bbox_expanded_by_dominated_points"] is False


def test_dominated_reference_point_expansion_is_diagnosed_not_corrected():
    ref = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.5],
        [10.0, 10.0],
    ])

    with pytest.warns(UserWarning, match="reference bbox is expanded"):
        result = _hv(GOOD, ref=ref)

    diagnostics = result.diagnostics
    assert np.array_equal(diagnostics["nbox_ideal"], [0.0, 0.0])
    assert np.array_equal(diagnostics["nbox_nadir"], [10.0, 10.0])
    assert np.array_equal(diagnostics["nd_ideal"], [0.0, 0.0])
    assert np.array_equal(diagnostics["nd_nadir"], [1.0, 1.0])
    assert np.array_equal(diagnostics["range_inflation"], [10.0, 10.0])
    assert diagnostics["max_range_inflation"] == pytest.approx(10.0)
    assert diagnostics["reference_points"] == 4
    assert diagnostics["reference_nd_points"] == 3
    assert diagnostics["reference_dominated_fraction"] == pytest.approx(0.25)
    assert diagnostics["bbox_expanded_by_dominated_points"] is True

    expected = evaluator.GEN_hypervolume([GOOD], 2, np.zeros(2), np.full(2, 10.0)).evaluate()[0]
    assert float(result) == pytest.approx(expected)


def test_outside_nbox_but_inside_bbox_is_recorded_and_can_contribute():
    front = np.array([[1.05, 0.5]])

    result = _hv(front, ref=np.array([[0.0, 1.0], [1.0, 0.0]]))

    assert result.diagnostics["outside_nbox_fraction"][0] == pytest.approx(1.0)
    assert result.diagnostics["outside_bbox_fraction"][0] == pytest.approx(0.0)
    assert float(result) > 0.0


def test_all_points_beyond_bbox_emit_floor_saturation_warning():
    front = np.array([[1.2, 0.5]])

    with pytest.warns(UserWarning, match="Hypervolume floor saturation"):
        result = _hv(front, ref=np.array([[0.0, 1.0], [1.0, 0.0]]))

    assert result.diagnostics["outside_bbox_fraction"][0] == pytest.approx(1.0)
    assert result.diagnostics["all_points_outside_bbox"][0]


def test_partial_bbox_saturation_is_recorded_without_floor_warning():
    front = np.array([[1.2, 0.5], [0.5, 0.5]])

    with warnings_not_matching("floor saturation"):
        result = _hv(front, ref=np.array([[0.0, 1.0], [1.0, 0.0]]))

    assert result.diagnostics["outside_bbox_fraction"][0] == pytest.approx(0.5)
    assert not result.diagnostics["all_points_outside_bbox"][0]


def test_better_than_ideal_is_recorded_without_clipping():
    front = np.array([[-0.2, 0.5]])

    result = _hv(front, ref=np.array([[0.0, 1.0], [1.0, 0.0]]))

    assert result.diagnostics["better_than_ideal_fraction"][0] == pytest.approx(1.0)
    assert float(result) > 1.1 * 0.6


def test_diagnostics_are_scale_independent(monkeypatch):
    class Mop:
        name = "TEST_MOP"

    class Front(np.ndarray):
        pass

    exp = GOOD.view(Front)
    exp.mop = Mop()
    baselines = importlib.import_module("moeabench.diagnostics.baselines")
    monkeypatch.setattr(
        baselines,
        "load_offline_baselines",
        lambda: {"_gt_registry": {"TEST_MOP__M2": REFERENCE.tolist()}},
    )

    results = [_hv(exp, ref=REFERENCE, scale=scale) for scale in ("raw", "rel", "abs")]
    keys = (
        "nbox_ideal",
        "nbox_nadir",
        "bbox_reference_point",
        "range_inflation",
        "outside_bbox_fraction",
        "raw_hv_fraction_of_nominal_bbox",
    )
    for key in keys:
        assert all(np.array_equal(results[0].diagnostics[key], result.diagnostics[key]) for result in results[1:])


def test_motivating_pooled_reference_is_diagnosed_without_hv_correction():
    exp1 = GOOD
    exp2 = np.vstack([GOOD * 1.5, [10.0, 10.0]])
    reference = [exp1, exp2]

    with pytest.warns(UserWarning, match="reference bbox is expanded"):
        hv1 = _hv(exp1, ref=reference)
    with pytest.warns(UserWarning, match="reference bbox is expanded"):
        hv2 = _hv(exp2, ref=reference)

    reference_all = np.vstack(reference)
    ideal = np.min(reference_all, axis=0)
    nadir = np.max(reference_all, axis=0)
    expected1 = evaluator.GEN_hypervolume([exp1], 2, ideal, nadir).evaluate()[0]
    expected2 = evaluator.GEN_hypervolume([exp2], 2, ideal, nadir).evaluate()[0]
    assert float(hv1) == pytest.approx(expected1)
    assert float(hv2) == pytest.approx(expected2)
    assert hv1.diagnostics["bbox_expanded_by_dominated_points"] is True
    assert np.all(hv1.diagnostics["range_inflation"] > 1.0)
    assert np.array_equal(hv1.diagnostics["nd_nadir"], np.array([0.8, 0.8]))


def test_self_referenced_hypervolume_has_no_external_diagnostics():
    result = _hv(GOOD)

    assert result.diagnostics == {}


def test_metric_matrix_slicing_preserves_diagnostics_object():
    result = _hv(GOOD, ref=REFERENCE)

    assert result[0].diagnostics is result.diagnostics


def test_near_ceiling_hv_is_reported_without_subjective_warning():
    front = np.array([[0.001, 0.001]])
    ref = np.array([[0.0, 1.0], [1.0, 0.0]])

    with warnings_not_matching("compress"):
        result = _hv(front, ref=ref)

    assert result.diagnostics["raw_hv_fraction_of_nominal_bbox"][0] > 0.99


def test_external_report_includes_reference_geometry_diagnostics():
    result = _hv(GOOD, ref=REFERENCE)

    report = result.report(show=False, markdown=False)

    assert "Reference Geometry" in report
    assert "N-box ideal" in report
    assert "N-box nadir" in report
    assert "B-box reference point" in report
    assert "Range inflation" in report
    assert "HV/BBox (final)" in report
    assert "Outside nbox (final)" in report
    assert "Outside bbox (final)" in report


class warnings_not_matching:
    """Assert that warnings may occur, but none matches a forbidden phrase."""

    def __init__(self, phrase):
        self.phrase = phrase

    def __enter__(self):
        import warnings

        self._manager = warnings.catch_warnings(record=True)
        self._caught = self._manager.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc, traceback):
        result = self._manager.__exit__(exc_type, exc, traceback)
        assert not any(self.phrase in str(item.message) for item in self._caught)
        return result
