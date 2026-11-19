# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent tests for issue #119; no existing regressions changed."""

import numpy as np
import pytest

from scripts.audit.gt_fps_preimage_study import (
    analytical_preimages,
    aggregate,
    compare_preimages,
)
from scripts.audit.gt_fps_comparison import farthest_candidates
from scripts.audit.gt_geometry_pilot import _MOPS


@pytest.mark.parametrize("name", ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"])
@pytest.mark.parametrize("m", [2, 3, 5])
@pytest.mark.parametrize("k", [2, 7])
def test_objective_space_fps_maps_back_to_exact_pareto_set(name, m, k):
    mop = _MOPS[name](M=m, K=k)
    front = farthest_candidates(name, m, max(12, m),
                                candidates=64, seed=13579)
    x = analytical_preimages(mop, front)
    assert x.shape == (len(front), mop.N)
    assert np.all((x >= 0.0) & (x <= 1.0))
    np.testing.assert_array_equal(x[:, m - 1:], 0.5 * np.ones((len(x), k)))
    evaluated = mop.evaluation(x)["F"]
    np.testing.assert_allclose(evaluated, front, atol=1e-10, rtol=0)


@pytest.mark.parametrize("name", ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"])
def test_boundaries_and_degenerate_preimages_are_valid(name):
    m = 5
    target = 0.5 if name == "DTLZ1" else 1.0
    mop = _MOPS[name](M=m)
    front = target * np.eye(m)
    x = analytical_preimages(mop, front)
    np.testing.assert_allclose(mop.evaluation(x)["F"], front,
                               atol=1e-10, rtol=0)


def test_repeated_controlled_comparison_preserves_seed_determinism():
    kwargs = dict(candidate_pool=96, reference_size=128, seed=57, k=3)
    a = compare_preimages("DTLZ1", 3, 24, **kwargs)
    b = compare_preimages("DTLZ1", 3, 24, **kwargs)
    a.pop("elapsed_seconds_diagnostic")
    b.pop("elapsed_seconds_diagnostic")
    assert a == b
    assert a["fps_preimage"]["n"] == 24
    assert a["preimage_max_evaluation_error"] < 1e-10
    summary = aggregate([a, b])
    assert summary[0]["measures"][0]["pairs"] == 2
    assert summary[0]["preimage_max_error"] < 1e-10


def test_invalid_off_front_data_is_rejected_before_inversion():
    mop = _MOPS["DTLZ2"](M=3)
    with pytest.raises(ValueError, match="front"):
        analytical_preimages(mop, np.array([[0.5, 0.5, 0.5]]))
    with pytest.raises(ValueError, match="positive"):
        analytical_preimages(mop, np.array([[1.0, -0.1, 0.0]]))
