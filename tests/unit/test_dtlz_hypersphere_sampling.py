# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from moeabench.mops.DTLZ2 import DTLZ2
from moeabench.mops.DTLZ3 import DTLZ3
from moeabench.mops.DTLZ4 import DTLZ4
from moeabench.mops.DTLZ5 import DTLZ5


@pytest.mark.parametrize("mop_cls", [DTLZ2, DTLZ3, DTLZ4])
@pytest.mark.parametrize("M", [3, 5, 10])
def test_spherical_dtlz_sampling_is_uniform_and_reproducible(mop_cls, M):
    """Analytical GT should cover the positive hypersphere without angular bias."""
    mop = mop_cls(M=M)
    n_points = 4000

    ps_first = mop.ps(n_points=n_points)
    ps_second = mop.ps(n_points=n_points)
    np.testing.assert_array_equal(ps_first, ps_second)

    front = mop.evaluation(ps_first)["F"]

    assert front.shape == (n_points, M)
    assert np.all(front >= -1e-14)
    np.testing.assert_allclose(
        np.sum(front**2, axis=1),
        1.0,
        atol=1e-12,
        rtol=0.0,
    )

    # A surface-uniform sample is permutation-symmetric across objectives.
    # In particular E[f_i^2] = 1/M for every coordinate. The former
    # independent-uniform-angle sampler violates this strongly for M > 2.
    second_moments = np.mean(front**2, axis=0)
    np.testing.assert_allclose(
        second_moments,
        np.full(M, 1.0 / M),
        atol=0.02,
        rtol=0.0,
    )

    # Exact axis points keep analytical ideal/nadir extents represented.
    axes = np.eye(M)
    for axis in axes:
        distance = np.linalg.norm(front - axis, axis=1)
        assert np.min(distance) <= 2e-12


@pytest.mark.parametrize("M", [3, 5, 10])
def test_dtlz2_and_dtlz4_sample_the_same_objective_geometry(M):
    """DTLZ4 decision-space bias must not bias its analytical Pareto-front GT."""
    dtlz2 = DTLZ2(M=M)
    dtlz4 = DTLZ4(M=M)

    np.testing.assert_allclose(
        dtlz4.pf(n_points=1000),
        dtlz2.pf(n_points=1000),
        atol=2e-12,
        rtol=0.0,
    )


def test_dtlz5_preserves_legacy_degenerate_pareto_set_sampling():
    """Changing DTLZ2.ps() must not silently alter inherited DTLZ5 sampling."""
    mop = DTLZ5(M=5)
    n_points = 64

    expected = np.zeros((n_points, mop.N))
    rng = np.random.RandomState(42)
    expected[:, :mop.M-1] = rng.random((n_points, mop.M - 1))
    expected[:, mop.M-1:] = 0.5

    np.testing.assert_array_equal(mop.ps(n_points=n_points), expected)


def test_dtlz2_calibrate_uses_corrected_analytical_gt(tmp_path, monkeypatch):
    """Default calibration should receive the corrected analytical DTLZ2 GT."""
    from moeabench.diagnostics import calibration

    captured = {}

    def fake_generate_baselines(name, gt, k_grid):
        captured["name"] = name
        captured["gt"] = np.array(gt, copy=True)
        captured["k_grid"] = list(k_grid)
        return {}

    monkeypatch.setattr(calibration, "_generate_baselines", fake_generate_baselines)
    monkeypatch.setattr(calibration.base, "register_baselines", lambda *args, **kwargs: None)

    mop = DTLZ2(M=10)
    sidecar = tmp_path / "DTLZ2_M10.json"

    assert mop.calibrate(
        source_baseline=str(sidecar),
        force=True,
        k_values=[10],
    )

    gt = captured["gt"]
    assert gt.shape == (2000, 10)
    np.testing.assert_allclose(np.sum(gt**2, axis=1), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        np.mean(gt**2, axis=0),
        np.full(10, 0.1),
        atol=0.02,
        rtol=0.0,
    )
