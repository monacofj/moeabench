# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent GT-geometry pilot tests; historical regressions unchanged."""
import numpy as np
import pytest

from scripts.audit.gt_geometry_pilot import audit, independent_reference, geometry_stats


@pytest.mark.parametrize("name", ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"])
@pytest.mark.parametrize("m", [3, 5])
def test_independent_surface_reference_is_on_exact_objective_geometry(name, m):
    points = independent_reference(name, 128, m, seed=123)
    replay = independent_reference(name, 128, m, seed=123)
    np.testing.assert_array_equal(points, replay)
    assert points.shape == (128, m)
    assert np.isfinite(points).all()
    assert np.all(points >= 0)
    if name == "DTLZ1":
        np.testing.assert_allclose(points.sum(axis=1), 0.5, atol=1e-14)
    else:
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1, atol=1e-14)


@pytest.mark.parametrize("name", ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"])
def test_gt_pilot_reports_reproducible_nonnegative_coverage_without_changing_gt(name):
    first = audit(name, 3, 32, reference_size=128, seed=9876)
    second = audit(name, 3, 32, reference_size=128, seed=9876)
    assert first == second
    assert first["budget"] == first["n"] == 32
    assert first["unique_rows"] <= 32
    assert first["geometry_max_residual"] < 1e-11
    for metric in ("reference_to_front_mean", "reference_to_front_p95",
                   "reference_to_front_max", "front_to_reference_mean",
                   "nearest_neighbor_mean", "extrema_max_distance"):
        assert np.isfinite(first[metric])
        assert first[metric] >= 0.0
    assert (first["nearest_neighbor_cv"] is None or
            np.isfinite(first["nearest_neighbor_cv"]))


def test_geometry_stats_exact_reference_is_zero_bidirectional_coverage():
    points = np.array([[0.5, 0.0], [0.0, 0.5]])
    stats = geometry_stats(points, points, "DTLZ1")
    assert stats["reference_to_front_max"] == 0
    assert stats["front_to_reference_mean"] == 0
    assert stats["extrema_max_distance"] == 0


def test_geometry_stats_rejects_invalid_input():
    with pytest.raises(ValueError, match="matching"):
        geometry_stats(np.ones((3, 2)), np.ones((4, 3)), "DTLZ2")
    with pytest.raises(ValueError, match="finite"):
        geometry_stats(np.array([[0.5, 0], [np.nan, 0]]),
                       np.array([[0.5, 0], [0, 0.5]]), "DTLZ1")
