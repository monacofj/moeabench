# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent observed-behavior verification for pymoo (issue #63).

Do not replace existing mutation-policy, frozen-front or dependency regression
tests. This checks pymoo's executed operator, not merely configured fields.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymoo.core.population import Population
from pymoo.core.problem import Problem

from moeabench.moeas._operators_pymoo import standard_polynomial_mutation


def test_polynomial_mutation_observed_per_variable_gate():
    """Detect upstream interpretation drift of individual/variable gates."""
    variables = 20
    individuals = 4000
    initial = np.full((individuals, variables), 0.5, dtype=float)
    problem = Problem(n_var=variables, xl=np.zeros(variables),
                      xu=np.ones(variables))
    population = Population.new("X", initial.copy())
    operator = standard_polynomial_mutation(variables)

    # The test suite already resets NumPy's process-global RNG per test.
    observed = operator.do(problem, population, inplace=False).get("X")
    assert observed.shape == initial.shape
    assert np.isfinite(observed).all()
    assert np.all((observed >= 0.0) & (observed <= 1.0))

    # If pymoo mistakenly treats the 1/n_var probability as an individual
    # gate and THEN applies its variable gate, the observed rate collapses
    # toward 1/n_var**2. The 0.005 window is about 6.5 standard errors
    # for 80,000 independent Bernoulli gates at p=.05.
    altered = np.count_nonzero(observed != initial)
    rate = altered / initial.size
    assert rate == pytest.approx(1.0 / variables, abs=0.005)
