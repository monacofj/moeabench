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

    assert float(_hv(beyond_nadir, ref=REFERENCE)) == 0.0
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
