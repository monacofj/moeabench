# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent DTLZ3/4 continuous-HV research; no old regression edits."""

from math import gamma, pi

import numpy as np
import pytest

from scripts.audit.hv_dtlz234_equivalence_issue89 import (
    exact_dtlz234_hv,
    observed_front_check,
)


@pytest.mark.parametrize("m", (2, 3, 4, 5, 8))
def test_original_dtlz2_3_4_true_spherical_hv_formula(m):
    expected = 1.1**m - pi**(m/2) / (2**m * gamma(m/2 + 1))
    assert exact_dtlz234_hv(m) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("name,m", [
    (name, m) for name in ("DTLZ2", "DTLZ3", "DTLZ4") for m in (2, 3, 4)
])
def test_original_dtlz234_actual_mop_sampler_below_continuous_hv(name, m):
    evidence = observed_front_check(name, m)
    assert 0 < evidence["ratio_sample_to_true"] < 1
    assert evidence["M"] == m


def test_dtlz3_original_distance_minimum_equals_dtlz2():
    import moeabench as mb
    for m in (2, 3, 4):
        a = mb.mops.DTLZ2(M=m)
        b = mb.mops.DTLZ3(M=m)
        points = a.ps(n_points=64)
        np.testing.assert_allclose(
            a.evaluation(points)["F"], b.evaluation(points)["F"],
            rtol=0, atol=1e-10
        )


def test_dtlz4_original_angular_transform_is_full_range():
    # x -> x^100 is continuous, increasing, and maps [0,1] onto [0,1].
    # Therefore it changes density but does not restrict spherical PF support.
    x = np.linspace(0, 1, 1001)
    transformed = x**100
    assert transformed[0] == 0 and transformed[-1] == 1
    assert np.all(np.diff(transformed) >= 0)
    assert np.all((transformed >= 0) & (transformed <= 1))
