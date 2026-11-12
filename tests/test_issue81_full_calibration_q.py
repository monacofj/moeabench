# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused pure-method tests for issue #81 complete calibration sensitivity."""
import numpy as np
import pytest

from calibration.scripts.study_dpf5_real_q_full_calibrations_issue81 import (
    seed_triplet, q_from_raw, _ranking, pooled_candidate,
)
from moeabench.diagnostics.qscore import compute_q_wasserstein


def test_complete_calibration_seeds_are_disjoint_across_k_and_replicates():
    seen = set()
    for k in (40, 100, 200):
        for replicate in range(32):
            seeds = seed_triplet(k, replicate)
            assert len(set(seeds)) == 3
            assert not seen.intersection(seeds)
            seen.update(seeds)


def test_out_of_domain_replicates_rejected():
    for args in ((1, 0), (201, 0), (40, -1), (40, 10000)):
        with pytest.raises(ValueError):
            seed_triplet(*args)


def test_reference_correction_uses_same_wasserstein_mapping():
    raw = np.array([0.12, 0.51, 0.95, 1.22])
    epsilon = .09
    failure = np.linspace(0, 2.0, 200)
    actual = q_from_raw(raw, epsilon, failure)
    adjusted = np.maximum(0., raw - epsilon)
    expected = compute_q_wasserstein(adjusted, np.zeros_like(adjusted), failure)
    assert actual == expected
    assert np.isfinite(actual)
    assert 0 <= actual <= 1


def test_ordering_is_deterministic_and_sensitive_to_rank_reversal():
    assert _ranking([.9, .3, .5]) == (0, 2, 1)
    assert _ranking([.3, .9, .5]) == (1, 2, 0)
    assert _ranking([.5, .5]) == (0, 1)


def test_pooled_quantiles_are_monotone_and_reproducible():
    candidates = [
        {"epsilon": .1, "ecdf": np.linspace(0., 1., 200)},
        {"epsilon": .2, "ecdf": np.linspace(0., 2., 200)},
    ]
    e1, v1 = pooled_candidate(candidates)
    e2, v2 = pooled_candidate(candidates)
    assert e1 == e2
    assert e1 == pytest.approx(.15)
    assert np.array_equal(v1, v2)
    assert len(v1) == 200 and np.all(np.diff(v1) >= 0)


def test_pooled_quantiles_require_independent_input_count():
    with pytest.raises(ValueError):
        pooled_candidate([{"epsilon": .1, "ecdf": np.zeros(200)}])
