# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent occupancy contracts for DTLZ7 and DPF2 true PF regions."""
import numpy as np
import pytest

from moeabench.mops.DTLZ7 import DTLZ7
from moeabench.mops.DPF2 import DPF2
from moeabench.mops._dtlz7_pareto_regions import (
    dtlz7_record_intervals, sample_dtlz7_record_parameters,
)


def component_id_rows(positions):
    """Encode first/second analytical record interval independently."""
    (lo1, hi1), (lo2, hi2) = dtlz7_record_intervals()
    positions = np.atleast_2d(positions)
    assert np.all(((positions >= lo1 - 1e-13) &
                   (positions <= hi1 + 1e-13)) |
                  ((positions > lo2) & (positions <= hi2 + 1e-13)))
    return [sum(int(x > lo2) << j for j, x in enumerate(row))
            for row in positions]


@pytest.mark.parametrize("dim,budget", [
    (1, 2), (2, 4), (4, 128), (8, 128), (8, 512),
    (9, 128), (9, 512), (9, 2000),
])
def test_region_sampler_occupies_each_component_before_repetition(dim, budget):
    points = sample_dtlz7_record_parameters(
        budget, dim, seed=42, stratified=True)
    ids = component_id_rows(points)
    assert len(set(ids)) == min(budget, 2**dim)
    assert len(ids) == budget
    np.testing.assert_array_equal(
        points, sample_dtlz7_record_parameters(
            budget, dim, seed=42, stratified=True))


@pytest.mark.parametrize("m,budget", [
    (3, 128), (5, 128), (10, 128), (10, 512), (10, 2000),
])
def test_dtlz7_exact_disconnected_region_occupancy(m, budget):
    mop = DTLZ7(M=m)
    decision = mop.ps(n_points=budget)
    assert decision.shape == (budget, mop.N)
    assert np.all(decision[:, m-1:] == 0)
    ids = component_id_rows(decision[:, :m-1])
    assert len(set(ids)) == min(budget, 2**(m-1))
    objectives = mop.evaluation(decision)["F"]
    assert np.unique(objectives, axis=0).shape[0] == budget
    np.testing.assert_array_equal(decision, mop.ps(n_points=budget))


@pytest.mark.parametrize("m,d,budget", [
    (5, 2, 128), (5, 3, 128), (10, 9, 128),
    (10, 9, 512), (10, 9, 2000),
])
def test_dpf2_exact_disconnected_region_occupancy(m, d, budget):
    mop = DPF2(M=m, D=d)
    decision = mop.ps(n_points=budget)
    assert decision.shape == (budget, mop.N)
    ids = component_id_rows(decision[:, :d-1])
    assert len(set(ids)) == min(budget, 2**(d-1))
    assert np.unique(mop.evaluation(decision)["F"], axis=0).shape[0] == budget
    np.testing.assert_array_equal(decision, mop.ps(n_points=budget))


def test_old_unstratified_sampler_api_is_still_deterministic():
    a = sample_dtlz7_record_parameters(128, 3, seed=42)
    b = sample_dtlz7_record_parameters(128, 3, seed=42)
    np.testing.assert_array_equal(a, b)
