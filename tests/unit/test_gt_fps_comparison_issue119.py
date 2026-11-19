# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent tests for the read-only FPS experiment; no prior tests modified."""
import numpy as np
import pytest

from scripts.audit.gt_fps_comparison import farthest_candidates, compare


@pytest.mark.parametrize("name", ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"])
def test_fps_candidate_determinism_geometry_and_extrema(name):
    first = farthest_candidates(name, 3, 24, candidates=96, seed=123)
    again = farthest_candidates(name, 3, 24, candidates=96, seed=123)
    np.testing.assert_array_equal(first, again)
    assert first.shape == (24, 3)
    target = 0.5 if name == "DTLZ1" else 1.0
    np.testing.assert_array_equal(first[:3], target * np.eye(3))
    assert len(np.unique(first, axis=0)) == 24
    assert np.all(first >= 0)
    residuals = (first.sum(axis=1) if name == "DTLZ1"
                 else np.linalg.norm(first, axis=1))
    np.testing.assert_allclose(residuals, target, atol=1e-13)


def test_small_controlled_comparison_is_deterministic_except_timing():
    a = compare("DTLZ1", 3, 24, candidates=96, reference_size=128, seed=202)
    b = compare("DTLZ1", 3, 24, candidates=96, reference_size=128, seed=202)
    a.pop("elapsed_seconds_noncertifying")
    b.pop("elapsed_seconds_noncertifying")
    assert a == b
    assert a["incumbent"]["n"] == 24
    assert a["candidate_fps"]["unique_rows"] == 24
    assert a["candidate_fps"]["extrema_max_distance"] == 0


@pytest.mark.parametrize("m,n,c", [(1, 8, 16), (3, 2, 16), (3, 16, 8)])
def test_invalid_candidate_parameters_rejected(m, n, c):
    with pytest.raises(ValueError):
        farthest_candidates("DTLZ2", m, n, candidates=c, seed=23)
