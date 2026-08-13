# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from moeabench.metrics.GEN_mc_hypervolume import GEN_mc_hypervolume


FRONTS = [
    np.array([[0.2, 0.8], [0.8, 0.2]]),
    np.array([[0.1, 0.7], [0.7, 0.1]]),
]


def _mc(seed, **kwargs):
    return GEN_mc_hypervolume(
        FRONTS,
        2,
        np.zeros(2),
        np.ones(2),
        n_samples=20_000,
        seed=seed,
        **kwargs,
    ).evaluate()


def test_monte_carlo_is_reproducible_for_the_same_seed():
    assert np.array_equal(_mc(17), _mc(17))


def test_monte_carlo_seed_changes_the_sample():
    assert not np.array_equal(_mc(17), _mc(18))


def test_monte_carlo_reports_progress_per_front():
    completed = []

    _mc(17, progress_callback=lambda: completed.append(len(completed) + 1))

    assert completed == [1, 2]


def test_monte_carlo_agrees_with_known_two_dimensional_volume():
    front = [np.array([[0.5, 0.5]])]
    result = GEN_mc_hypervolume(
        front,
        2,
        np.zeros(2),
        np.ones(2),
        n_samples=100_000,
        seed=11,
    ).evaluate()[0]

    assert result == pytest.approx(0.36, abs=0.005)


@pytest.mark.parametrize("n_samples", [0, -1, 1.5])
def test_monte_carlo_rejects_invalid_sample_counts(n_samples):
    with pytest.raises(ValueError, match="positive integer"):
        GEN_mc_hypervolume(
            FRONTS,
            2,
            np.zeros(2),
            np.ones(2),
            n_samples=n_samples,
        )
