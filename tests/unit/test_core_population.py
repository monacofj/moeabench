# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from MoeaBench.core.run import Population

def test_population_creation():
    """Verify basic instantiation and shape validation."""
    objs = np.random.rand(10, 3)
    vars = np.zeros((10, 5))
    
    pop = Population(objs, vars)
    assert len(pop) == 10
    assert pop.objectives.shape == (10, 3)
    assert pop.variables.shape == (10, 5)

def test_population_algebra_slicing():
    """Verify slicing returns a new valid Population object."""
    objs = np.random.rand(20, 2)
    vars = np.zeros((20, 2))
    pop = Population(objs, vars)
    
    # Slice first 5
    sub_pop = pop[:5]
    
    assert isinstance(sub_pop, Population)
    assert len(sub_pop) == 5
    assert np.array_equal(sub_pop.objectives, objs[:5])

def test_population_algebra_getitem():
    """Verify integer indexing returns a single individual (View or Copy)."""
    objs = np.array([[0.1, 0.2], [0.3, 0.4]])
    vars = np.zeros((2, 2))
    pop = Population(objs, vars)
    
    ind = pop[1]
    # In MoeaBench, indexing might return a Population of size 1 or an individual struct.
    # Current implementation returns a Population of size 1 for consistency.
    assert isinstance(ind, Population)
    assert len(ind) == 1
    assert np.allclose(ind.objectives[0], [0.3, 0.4])

def test_population_algebra_concatenation():
    """Verify proper merging of two populations."""
    pop1 = Population(np.ones((10, 2)), np.zeros((10, 2)))
    pop2 = Population(np.zeros((5, 2)), np.zeros((5, 2)))
    
    # Assumes the __add__ operator is implemented or use a merge utility
    # Checking if MoeaBench.core.run.Population supports + operator
    try:
        merged = pop1 + pop2
        assert len(merged) == 15
        assert np.array_equal(merged.objectives[0], [1, 1])
        assert np.array_equal(merged.objectives[14], [0, 0])
    except TypeError:
        pytest.fail("Population class does not support '+' operator for concatenation.")

def test_population_conversion():
    """Verify internal conversion of list inputs to numpy."""
    obs_list = [[1, 2], [3, 4]]
    pop = Population(obs_list)
    
    assert isinstance(pop.objectives, np.ndarray)
    assert pop.objectives.shape == (2, 2)
