# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collected replacement for the numerical DPF3/EMD checks in debug_dpf3_emd.py."""

import numpy as np
import pytest

import moeabench as mb


def test_dpf3_emd_identical_front_and_known_translation():
    # DPF3.pf() filters dominated analytical candidates before returning GT.
    gt = np.asarray(mb.mops.DPF3().pf(n_points=250), dtype=np.float64)
    assert gt.ndim == 2 and gt.shape[1] == 3
    assert len(gt) > 0 and np.isfinite(gt).all()

    same = float(mb.metrics.emd(gt, ref=gt, progress=False))
    assert same == pytest.approx(0.0, abs=1e-12)

    # The implementation is the mean per-coordinate 1-D Wasserstein distance.
    # A uniform translation of every objective by delta has exact EMD=delta.
    delta = 1e-4
    shifted = float(mb.metrics.emd(gt + delta, ref=gt, progress=False))
    assert shifted == pytest.approx(delta, abs=1e-12)
