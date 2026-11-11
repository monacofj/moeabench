# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Issue #75 real-front experiment scoring contract and failure modes."""

import numpy as np
import pytest
from moeabench.diagnostics import baselines, qscore
from calibration.scripts.study_real_moea_q_sensitivity import (
    clinical_q_for_fixed_front,
)


def test_scoring_keeps_front_constant_and_corrects_observed_and_null():
    raw = np.linspace(0.01, 1.99, 43)
    blur = np.linspace(0.01, 2.01, 200)
    a, b = raw.copy(), blur.copy()
    for epsilon in (0.0, 0.03, 0.4, 0.9):
        value = clinical_q_for_fixed_front(raw, blur, epsilon, k=40)
        corrected = np.maximum(0, raw - epsilon)
        adjusted_blur = np.maximum(0, blur - epsilon)
        from unittest.mock import patch
        with patch.object(baselines, "snap_k", side_effect=lambda K, problem=None: K), \
             patch.object(baselines, "get_baseline_ecdf",
                          return_value=(0.0, float(np.median(adjusted_blur)), adjusted_blur)):
            oracle = float(qscore.q_closeness(
                corrected, problem="RealMOEAResidueSensitivity", k=40,
            ).value)
        assert value == pytest.approx(oracle, abs=1e-14)
        assert 0 <= value <= 1
    np.testing.assert_array_equal(raw, a)
    np.testing.assert_array_equal(blur, b)


def test_clinical_q_refuses_invalid_fronts_or_reference():
    good_front = np.linspace(0.1, 1.0, 32)
    good_ecdf = np.linspace(0.1, 2.0, 200)
    for dist, ecdf, eps in [
        ([], good_ecdf, 0),
        ([np.nan], good_ecdf, 0),
        ([-0.1], good_ecdf, 0),
        (good_front, [0.1]*100, 0),
        (good_front, good_ecdf, -0.1),
        (good_front, good_ecdf, np.inf),
    ]:
        with pytest.raises(ValueError, match="finite nonnegative"):
            clinical_q_for_fixed_front(dist, ecdf, eps, k=40)


def test_identical_good_and_bad_distributions_are_zero_q():
    raw = np.linspace(0.1, 2.0, 200)
    for eps in (0, 0.1, 1):
        assert clinical_q_for_fixed_front(raw, raw, eps, k=40) == pytest.approx(0)
