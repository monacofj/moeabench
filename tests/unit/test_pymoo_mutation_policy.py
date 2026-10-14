# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the shared pymoo mutation policy."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from pymoo.core.population import Population
from pymoo.core.problem import Problem

from moeabench.mops.DTLZ2 import DTLZ2
from moeabench.moeas._operators_pymoo import standard_polynomial_mutation


@pytest.mark.dependency_verify
def test_standard_polynomial_mutation_preserves_per_variable_rate():
    """pymoo >= 0.6 must apply 1/n_var at the variable, not individual, gate."""
    n_var = 20
    mutation = standard_polynomial_mutation(n_var)

    assert mutation.prob.value == pytest.approx(1.0)
    assert mutation.prob_var.value == pytest.approx(1.0 / n_var)
    assert mutation.eta.value == pytest.approx(20.0)


@pytest.mark.dependency_verify
def test_standard_polynomial_mutation_observed_rate_is_per_variable():
    """Exercise pymoo itself so semantic drift cannot hide behind attributes."""
    n_var = 20
    n_individuals = 4000
    expected_rate = 1.0 / n_var

    problem = Problem(
        n_var=n_var,
        xl=np.zeros(n_var),
        xu=np.ones(n_var),
    )
    original = np.full((n_individuals, n_var), 0.5)
    population = Population.new("X", original.copy())
    mutation = standard_polynomial_mutation(n_var)

    mutated = mutation.do(
        problem,
        population,
        inplace=False,
        random_state=np.random.default_rng(12345),
    ).get("X")

    observed_rate = np.count_nonzero(mutated != original) / original.size
    assert observed_rate == pytest.approx(expected_rate, abs=0.005)


@pytest.mark.dependency_verify
@pytest.mark.parametrize(
    "module_name, engine_name, algorithm_name",
    [
        ("moeabench.moeas._moead_pymoo", "MOEAD_pymoo", "MOEAD"),
        ("moeabench.moeas._nsga_pymoo", "NSGA_pymoo", "NSGA3"),
        ("moeabench.moeas._unsga_pymoo", "UNSGA_pymoo", "UNSGA3"),
        ("moeabench.moeas._rvea_pymoo", "RVEA_pymoo", "RVEA"),
        ("moeabench.moeas._spea_pymoo", "SPEA_pymoo", "SPEA2"),
    ],
)
def test_pymoo_wrappers_use_shared_mutation_policy(
    monkeypatch, module_name, engine_name, algorithm_name
):
    """Every affected wrapper must obtain mutation through the shared policy helper."""
    module = importlib.import_module(module_name)
    sentinel_mutation = object()
    captured = {}

    def fake_mutation(n_var):
        captured["n_var"] = n_var
        return sentinel_mutation

    class FakeAlgorithm:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(module, "standard_polynomial_mutation", fake_mutation)
    monkeypatch.setattr(module, "get_reference_directions", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, algorithm_name, FakeAlgorithm)

    mop = DTLZ2(M=3)
    engine = getattr(module, engine_name)(
        mop,
        population=24,
        generations=1,
        seed=12345,
    )

    def capture_algorithm(algorithm):
        captured["algorithm"] = algorithm
        return None

    engine.run_minimize = capture_algorithm
    engine.evaluation()

    assert captured["n_var"] == engine.Nvar
    assert captured["algorithm"].kwargs["mutation"] is sentinel_mutation
