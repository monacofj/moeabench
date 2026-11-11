# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Issue #75: independent reproducibility tests for DEAP-backed NSGA-II.

The clinical audit should never be validated with an experiment whose declared
seed does not actually control all evolutionary operators. The tests here
exercise two independent complete runs, not merely the initialization RNG.
"""

import random

import numpy as np
import moeabench as mb


def _dtlz1_front(seed):
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ1(M=3)
    exp.moea = mb.moeas.NSGA2(population=20, generations=12)
    exp.run(seed=seed, silent=True)
    return np.asarray(exp.front(-1)).copy()


def test_nsga2_same_seed_repeats_full_evolution_despite_global_rng_noise():
    """Same seed must produce the same front after stochastic evolution."""
    a = _dtlz1_front(175)
    # Deliberately change module-global RNG between the two runs. Previously
    # the DEAP operators used that state even when a seed was requested.
    random.seed(9921)
    for _ in range(101):
        random.random()
    b = _dtlz1_front(175)
    np.testing.assert_array_equal(a, b)


def test_nsga2_different_seeds_produce_different_evolution():
    a = _dtlz1_front(175)
    b = _dtlz1_front(176)
    assert not np.array_equal(a, b)


def test_nsga2_private_deap_calls_restore_global_rng_state():
    """A private DEAP stochastic operator must not leak into global random."""
    engine = mb.moeas.NSGA2(population=20, generations=1, seed=175)
    engine.rng = random.Random(175)
    random.seed(9)
    before = random.getstate()
    draw = lambda: (random.random(), random.random(), random.random())
    first = engine._with_private_deap_rng(draw)
    assert random.getstate() == before
    engine.rng = random.Random(175)
    assert engine._with_private_deap_rng(draw) == first
    assert random.getstate() == before
