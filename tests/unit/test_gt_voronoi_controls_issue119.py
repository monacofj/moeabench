# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New unit coverage for exploratory Voronoi study (#119), old tests untouched."""
import numpy as np
import pytest
from scripts.audit.gt_geometry_pilot import _MOPS, independent_reference
from scripts.audit.gt_voronoi_controls_issue119 import (
    with_analytical_extrema, projected_mc_cvt, study, summarize,
)
from scripts.audit.gt_fps_preimage_study import analytical_preimages


@pytest.mark.parametrize("name", ["DTLZ1", "DTLZ2"])
@pytest.mark.parametrize("m", [3, 5])
def test_extrema_control_preserves_budget_and_incumbent_other_points(name, m):
    original = _MOPS[name](M=m).pf(n_points=24)
    anchored = with_analytical_extrema(original, name)
    np.testing.assert_array_equal(anchored[m:], original[m:])
    np.testing.assert_array_equal(
        anchored[:m], (0.5 if name == "DTLZ1" else 1.0) * np.eye(m))
    assert anchored.shape == original.shape
    assert len(np.unique(anchored, axis=0)) == len(anchored)


@pytest.mark.parametrize("name", ["DTLZ1", "DTLZ2"])
def test_projected_voronoi_is_deterministic_and_analytically_valid(name):
    m = 3
    mop = _MOPS[name](M=m, K=3)
    old = with_analytical_extrema(mop.pf(n_points=18), name)
    pool = independent_reference(name, 72, m, seed=123)
    a = projected_mc_cvt(name, old, pool, iterations=5)
    b = projected_mc_cvt(name, old, pool, iterations=5)
    np.testing.assert_array_equal(a, b)
    assert len(a) == len(old)
    np.testing.assert_array_equal(
        a[:m], (0.5 if name == "DTLZ1" else 1.0) * np.eye(m))
    assert np.all(a >= 0)
    geometric = a.sum(axis=1) if name == "DTLZ1" else np.linalg.norm(a, axis=1)
    np.testing.assert_allclose(geometric, 0.5 if name == "DTLZ1" else 1.0,
                               rtol=0, atol=1e-12)
    x = analytical_preimages(mop, a)
    np.testing.assert_allclose(mop.evaluation(x)["F"], a, atol=1e-10, rtol=0)


def test_real_mop_experiment_contains_all_five_method_variants():
    opts = dict(pool_size=80, reference_size=96, seed=21, iterations=3)
    a = study("DTLZ1", 3, 16, **opts)
    b = study("DTLZ1", 3, 16, **opts)
    assert a["stats"] == b["stats"]
    assert set(a["stats"]) == {
        "incumbent", "anchored", "fps", "anchored_mc_cvt", "fps_mc_cvt"}
    assert max(a["inverse_errors"].values()) < 1e-10
    assert summarize([a, b])[0]["seeds"] == 2


def test_reject_bad_dimensions():
    with pytest.raises(ValueError, match="n>=M"):
        with_analytical_extrema(np.ones((2, 3)), "DTLZ1")
    with pytest.raises(ValueError, match="Require"):
        projected_mc_cvt("DTLZ1", np.ones((3, 3)), np.ones((2, 3)))
