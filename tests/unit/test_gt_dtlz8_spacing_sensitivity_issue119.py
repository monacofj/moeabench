# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Positive independent controls for DTLZ8 spacing/reference sensitivity."""
import numpy as np
import pytest

from scripts.audit.gt_dtlz8_candidate_issue119 import candidate_objectives
from scripts.audit.gt_dtlz8_spacing_sensitivity_issue119 import (
    _component_indices, component_spacing, experiment, finite_igd_plus,
)


def test_igd_plus_direction_and_exact_reference():
    reference = np.array([[0.2, 0.8], [0.8, 0.2]])
    np.testing.assert_allclose(finite_igd_plus(reference, reference), 0.0)
    # Approximations are worse in both objectives: strictly positive IGD+.
    bad = reference + 0.1
    np.testing.assert_allclose(finite_igd_plus(reference, bad), 0.1)
    # A better approximation cannot increase positive-part IGD+.
    assert finite_igd_plus(reference, reference - 0.05) == 0.0


@pytest.mark.parametrize("m", [2, 3, 4, 5, 10])
def test_component_labels_and_budget_for_analytical_candidate(m):
    f = candidate_objectives(m, 128, patch_strategy="lattice")
    labels = _component_indices(f)
    assert np.count_nonzero(labels == -1) == 0
    assert set(labels) == set(range(m if m > 2 else 1))
    spacing = component_spacing(f)
    assert spacing["off_pareto"]["rows"] == 0
    assert spacing["line"]["points"] >= 2
    assert np.isfinite(spacing["line"]["nn_cv"])


def test_component_spacing_excludes_off_front_points():
    f = candidate_objectives(3, 64)
    wrong = f[0].copy()
    wrong[0] += 0.1
    spacing = component_spacing(np.vstack((f, wrong)))
    assert spacing["off_pareto"]["rows"] == 1


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_igd_sensitivity_experiment_has_explicit_references_and_populations(m):
    r = experiment(m, 128, seed=11941, population_per_region=8)
    assert r["budget"] == 128
    assert not r["canonical_mixed_dimensional_measure_defined"]
    assert not r["certified_continuous_covering_bound"]
    assert set(("incumbent", "halton", "lattice")).issubset(
        r["igd_plus_by_finite_reference_and_population"])
    for scores in r["igd_plus_by_finite_reference_and_population"].values():
        assert all(np.isfinite(list(scores.values())))
        assert all(v >= 0 for v in scores.values())
    if m == 4:
        assert "preserving_completion" in r["reference_spacing"]
    else:
        assert "preserving_completion" not in r["reference_spacing"]


def test_igd_plus_explicit_reference_changes_numerical_value():
    # Same independent population, two different finite samples of the true PF.
    population = candidate_objectives(4, 32, patch_strategy="lattice")
    first = candidate_objectives(4, 64)
    second = candidate_objectives(4, 64, patch_strategy="lattice")
    assert not np.isclose(
        finite_igd_plus(first, population),
        finite_igd_plus(second, population), rtol=1e-8, atol=1e-10)
