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


def test_all_controller_iteration_streams_are_disjoint_from_ideal_and_blur():
    # Regression: old triplets (base, base+1, base+2) collided at fit it=1
    # and even with the next replicate's seeds at later iterations.
    seen = set()
    for k in range(2, 201):
        for replicate in (0, 1, 2, 9998, 9999):
            ideal, fit, blur = seed_triplet(k, replicate)
            consumed = {ideal, *range(fit, fit + 6), blur}
            assert len(consumed) == 8
            assert not seen.intersection(consumed)
            seen.update(consumed)


def test_complete_replicate_held_out_draw_uses_no_controller_stream(monkeypatch):
    from scipy.spatial import cKDTree
    import calibration.scripts.study_dpf5_real_q_full_calibrations_issue81 as module
    from calibration.scripts.validate_dpf5_closeness_protocol_issue81 import frozen_reference
    from calibration.scripts.generate_baselines import estimate_gt_normals
    _, gt, ideal, nadir, model = frozen_reference()
    consumed = []
    original = np.random.RandomState

    class spy(original):
        def __init__(self, seed):
            consumed.append(seed)
            super().__init__(seed)

    monkeypatch.setattr(np.random, 'RandomState', spy)
    record = module.make_replicate(model=model, gt=gt, ideal=ideal, nadir=nadir,
                                  tree=cKDTree(gt), normals=estimate_gt_normals(gt),
                                  k=100, scale=.09085520863957625,
                                  replicate=0, ideal_size=256)
    assert len(consumed) >= 3  # fitting actually adjusts sigma
    assert len(set(consumed)) == len(consumed)
    assert consumed[-1] == record['seeds']['blur']
    assert set(consumed[:-1]).issubset(record['fit_reserved_seeds'])


def test_pairwise_analysis_distinguishes_ties_and_strict_reversals():
    from calibration.scripts.analyze_dpf5_real_q_issue81 import pair_analysis
    rows = [
        dict(candidate_q=[.5, .5, .6], historical_q=.4, pooled_reference_q_experimental=.55),
        dict(candidate_q=[.5, .6, .5], historical_q=.5, pooled_reference_q_experimental=.55),
    ]
    pair, = pair_analysis(rows)
    assert pair['exact_tie_columns'] == 1
    assert pair['strict_reversal_across_calibrations']
    assert pair['strict_reversal_vs_historical']
    assert not pair['strict_reversal_pool_vs_historical']
