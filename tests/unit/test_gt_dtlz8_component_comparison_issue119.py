# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke checks for independent component-level DTLZ8 diagnostics."""
import numpy as np
import pytest

from scripts.audit.gt_dtlz8_component_comparison_issue119 import (
    _distance_statistics, compare, independent_probes,
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
