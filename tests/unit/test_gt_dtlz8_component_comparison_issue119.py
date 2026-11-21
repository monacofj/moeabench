# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke checks for independent component-level DTLZ8 diagnostics."""
import numpy as np
import pytest

from scripts.audit.gt_dtlz8_component_comparison_issue119 import (
    _distance_statistics, compare, independent_probes,
    m4_preserving_completion,
)


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_dtlz8_independent_reference_probe_components(m):
    probes = independent_probes(m, n_per_component=17, offset=11941)
    assert len(probes) == (m if m >= 3 else 1)
    for component in probes.values():
        assert component.shape == (17, m)
        assert np.all((component >= 0) & (component <= 1))


def test_dtlz8_component_diagnostics_identical_sets_zero():
    probes = independent_probes(3, n_per_component=12, offset=123)
    x = probes["patch_0"]
    report = _distance_statistics(x, x)
    assert report["max_euclidean_gap"] < 1e-12
    assert report["mean_euclidean_gap"] < 1e-12
    assert report["mean_igd_plus_positive_part"] < 1e-12


@pytest.mark.parametrize("m", [2, 3, 4])
def test_dtlz8_component_comparison_schema(m):
    report = compare(m, 32, probes_per_component=16, offset=143)
    assert report["candidate_count"] == 32
    assert report["candidate_unique"] == 32
    assert report["lattice_count"] == 32
    assert report["lattice_unique"] == 32
    assert not report["continuous_coverage_certified"]
    assert not report["canonical_mixed_dimensional_measure_chosen"]
    for scores in report["components"].values():
        for method in ("incumbent", "candidate", "lattice"):
            assert all(np.isfinite(list(scores[method].values())))


def test_dtlz8_probe_seeds_are_distinct_and_reproducible():
    first = independent_probes(4, n_per_component=128, offset=11941)
    replay = independent_probes(4, n_per_component=128, offset=11941)
    other = independent_probes(4, n_per_component=128, offset=11942)
    for component in first:
        np.testing.assert_array_equal(first[component], replay[component])
        assert not np.array_equal(first[component], other[component])


def test_dtlz8_m4_completion_preserves_all_incumbent_points():
    from moeabench.mops.DTLZ8 import DTLZ8
    from scripts.audit.gt_dtlz8_candidate_issue119 import candidate_objectives

    budget = 128
    incumbent = DTLZ8(M=4).pf(n_points=budget)
    lattice = candidate_objectives(4, budget, patch_strategy="lattice")
    completed = m4_preserving_completion(incumbent, lattice, budget)
    assert completed.shape == (budget, 4)
    assert len(np.unique(completed, axis=0)) == budget
    for point in np.unique(incumbent, axis=0):
        assert np.any(np.all(completed == point, axis=1))


@pytest.mark.parametrize("budget", [128, 512])
def test_dtlz8_m4_completion_never_worsens_independent_probe_gaps(budget):
    result = compare(4, budget, probes_per_component=64, offset=11942)
    assert result["m4_completion_count"] == budget
    assert result["m4_completion_unique"] == budget
    for component in result["components"].values():
        for metric in ("max_euclidean_gap", "mean_euclidean_gap",
                       "mean_igd_plus_positive_part"):
            assert component["m4_completion"][metric] <= (
                component["incumbent"][metric] + 1e-13)


def test_dtlz8_m4_completion_rejects_off_front_data():
    from scripts.audit.gt_dtlz8_candidate_issue119 import candidate_objectives
    invalid = np.array([[0.2, 0.2, 0.2, 0.5]])
    lattice = candidate_objectives(4, 32, patch_strategy="lattice")
    with pytest.raises(ValueError, match="off-Pareto"):
        m4_preserving_completion(invalid, lattice, 32)
