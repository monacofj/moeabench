# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent tests for ADR 0058 scalar absolute HV reference policy.

Existing regression tests and historical reference data remain untouched.
"""
import json
from math import factorial, gamma, pi

import numpy as np
import pytest
import moeabench as mb


class OwnedFront(np.ndarray):
    """A front carrying its actual owning MOP."""


def owned(front, mop):
    result = np.asarray(front, dtype=float).copy().view(OwnedFront)
    result.mop = mop
    return result


def hv(front, ref, *, scale="abs", mode="exact"):
    return mb.metrics.hv(
        front, ref=ref, scale=scale, mode=mode,
        n_samples=20000, mc_seed=17, progress=False
    )


def expected_true_hv(name, m):
    if name == "DTLZ1":
        return 1.1**m - 1/factorial(m)
    return 1.1**m - pi**(m/2) / (2**m * gamma(m/2 + 1))


@pytest.mark.parametrize("name,m", [
    ("DTLZ1", 2), ("DTLZ1", 3), ("DTLZ1", 4),
    ("DTLZ2", 2), ("DTLZ2", 3), ("DTLZ2", 4),
    ("DTLZ3", 2), ("DTLZ3", 3), ("DTLZ3", 4),
    ("DTLZ4", 2), ("DTLZ4", 3), ("DTLZ4", 4),
])
def test_analytical_calibration_and_true_geometry_use_exact_reference(
    name, m, tmp_path
):
    mop = getattr(mb.mops, name)(M=m)
    assert mop.calibrate(
        source_baseline=tmp_path / "analytical.json", n_points=64
    )
    reference = mop.calibrated_gt
    output = hv(owned(reference, mop), reference)
    expected = expected_true_hv(name, m)
    assert output.diagnostics["hv_abs_reference_basis"] == "continuous_exact"
    assert output.diagnostics["hv_abs_denominator"] == pytest.approx(expected)
    assert output.diagnostics["hv_abs_gt_points"] == len(reference)
    # A finite GT is a strict under-approximation in these tested cases.
    assert 0 < float(output) < 1


def test_dpf1_m3_d2_uses_proved_degenerate_continuous_denominator(tmp_path):
    mop = mb.mops.DPF1(M=3, D=2)
    mop.calibrate(source_baseline=tmp_path / "dpf.json", n_points=64)
    gt = mop.calibrated_gt
    result = hv(owned(gt, mop), gt)
    expected = (1.1**3 - .1**3)/3 + .1*1.1**2
    assert result.diagnostics["hv_abs_reference_basis"] == "continuous_exact"
    assert result.diagnostics["hv_abs_denominator"] == pytest.approx(expected)
    assert 0 < float(result) < 1


def test_external_custom_gt_preserves_historical_12_over_7_regression(tmp_path):
    gt = np.array([[0., .5], [.5, 0.]])
    evaluated = np.array([[.25, .25]])
    mop = mb.mops.DTLZ1(M=2)
    mop.calibrate(source_baseline=tmp_path / "custom.json", source_gt=gt)
    result = hv(owned(evaluated, mop), gt)
    assert result.diagnostics["hv_abs_reference_basis"] == "sampled_gt"
    assert result.diagnostics["hv_abs_denominator"] == pytest.approx(.21)
    assert float(result) == pytest.approx(12/7)
    assert float(hv(evaluated, gt, scale="rel")) == pytest.approx(12/7)
    assert float(hv(evaluated, gt, scale="raw")) == pytest.approx(.36)


def test_calibration_load_restores_analytical_provenance(tmp_path):
    path = tmp_path / "gt.json"
    original = mb.mops.DTLZ2(M=2)
    original.calibrate(source_baseline=path, n_points=64)
    loaded = mb.mops.DTLZ2(M=2)
    assert loaded.calibrate(source_baseline=path) is False
    assert loaded._gt_provenance == "analytical"
    gt = loaded.calibrated_gt
    result = hv(owned(gt, loaded), gt)
    assert result.diagnostics["hv_abs_reference_basis"] == "continuous_exact"


def test_legacy_sidecar_missing_gt_source_uses_sampled_fallback(tmp_path):
    path = tmp_path / "legacy.json"
    gt = np.array([[0., .5], [.5, 0.]])
    path.write_text(json.dumps({
        "problem_id": "DTLZ1", "mop_dimension": 2,
        "gt_reference": gt.tolist(),
    }), encoding="utf-8")
    mop = mb.mops.DTLZ1(M=2)
    assert mop.calibrate(source_baseline=path) is False
    assert mop._gt_provenance is None
    result = hv(owned([[.25, .25]], mop), gt)
    assert result.diagnostics["hv_abs_reference_basis"] == "sampled_gt"
    assert float(result) == pytest.approx(12/7)


def test_shifted_external_normalization_falls_back_to_sampled_gt(tmp_path):
    mop = mb.mops.DTLZ1(M=2)
    mop.calibrate(source_baseline=tmp_path / "gt.json", n_points=64)
    gt = mop.calibrated_gt
    other_bounds = np.vstack([gt, [[-.1, .6]]])
    result = hv(owned(gt, mop), other_bounds)
    assert result.diagnostics["hv_abs_reference_basis"] == "sampled_gt"
    assert float(result) == pytest.approx(1)


def test_subclass_is_not_assumed_to_have_proved_original_geometry(tmp_path):
    class AlteredMOP(mb.mops.DTLZ2):
        pass

    mop = AlteredMOP(M=2)
    mop.calibrate(source_baseline=tmp_path / "subclass.json", n_points=64)
    gt = mop.calibrated_gt
    result = hv(owned(gt, mop), gt)
    assert result.diagnostics["hv_abs_reference_basis"] == "sampled_gt"
    assert float(result) == pytest.approx(1)


def test_fast_backend_returns_scalar_and_exact_denominator(tmp_path):
    mop = mb.mops.DTLZ2(M=2)
    mop.calibrate(source_baseline=tmp_path / "gt.json", n_points=64)
    gt = mop.calibrated_gt
    result = hv(owned(gt, mop), gt, mode="fast")
    assert result.diagnostics["hv_abs_reference_basis"] == "continuous_exact"
    assert result.diagnostics["hv_abs_denominator"] == pytest.approx(
        expected_true_hv("DTLZ2", 2)
    )
    assert np.isscalar(float(result))
    assert np.isfinite(float(result))
