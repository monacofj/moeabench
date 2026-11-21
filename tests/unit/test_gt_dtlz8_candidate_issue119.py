# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent scientific checks of the proposed DTLZ8 sample."""
from itertools import combinations

import numpy as np
import pytest

from moeabench.mops.DTLZ8 import DTLZ8
from scripts.audit.gt_dtlz8_candidate_issue119 import (
    candidate_decisions, candidate_objectives,
)


@pytest.mark.parametrize("m,budget", [
    (3, 128), (4, 128), (5, 512), (10, 512),
    (3, 2000), (10, 2000),
])
def test_dtlz8_candidate_exact_count_and_objective_mapping(m, budget):
    f = candidate_objectives(m, budget)
    x = candidate_decisions(m, budget)
    mop = DTLZ8(M=m)
    np.testing.assert_allclose(mop.evaluation(x)["F"], f, rtol=0, atol=2e-15)
    assert f.shape == (budget, m)
    assert np.unique(f, axis=0).shape[0] == budget
    assert np.isfinite(f).all()
    assert ((f >= 0) & (f <= 1)).all()

    # Independent feasibility reconstruction from the published
    # objective constraints, without calling mop._calc_constraints().
    t = f[:, -1]
    assert np.min(t[:, None] + 4 * f[:, :-1] - 1) >= -2e-14
    pairs = np.array([f[:, i] + f[:, j]
                     for i, j in combinations(range(m - 1), 2)])
    assert np.min(2 * t + pairs.min(axis=0) - 1) >= -2e-14


@pytest.mark.parametrize("m", [3, 4, 5, 10])
def test_dtlz8_candidate_covers_line_and_every_patch(m):
    f = candidate_objectives(m, 512)
    t = f[:, -1]
    line = t >= 1 / 3
    assert np.count_nonzero(line) >= 2
    np.testing.assert_allclose(f[line, :-1], ((1-t[line])/4)[:, None],
                               rtol=0, atol=2e-15)
    assert np.any(np.isclose(f[line, -1], 1.0, atol=0, rtol=0))
    for j in range(m-1):
        patch = f[~line]
        lowest = np.argmin(patch[:, :-1], axis=1)
        selected = patch[lowest == j]
        assert len(selected) > 0
        if m > 3:
            # The other M-2 coordinates are identical on each patch.
            other = np.delete(selected[:, :-1], j, axis=1)
            np.testing.assert_allclose(
                other, np.repeat(other[:, :1], other.shape[1], axis=1),
                rtol=0, atol=2e-15)


def test_dtlz8_candidate_deterministic_and_rejects_unsupported_geometry():
    np.testing.assert_array_equal(candidate_objectives(4, 128),
                                  candidate_objectives(4, 128))
    with pytest.raises(ValueError, match="M>=3"):
        candidate_objectives(2, 128)
    with pytest.raises(ValueError, match="M\\+1"):
        candidate_objectives(5, 5)
