# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Low-budget end-to-end empirical GT calibration (public source_search)."""

import json

import numpy as np

import moeabench as mb


def test_empirical_gt_search_with_real_moea(tmp_path):
    mop = mb.mops.DTLZ2(M=2)
    moea = mb.moeas.NSGA2deap(population=12, generations=2, seed=7)
    sidecar = tmp_path / "empirical-gt.json"

    assert mop.calibrate(source_search=moea, source_baseline=sidecar) is True

    gt = mop.calibrated_gt
    assert gt.ndim == 2 and gt.shape[1] == 2
    assert len(gt) > 0
    assert np.isfinite(gt).all()

    persisted = json.loads(sidecar.read_text(encoding="utf-8"))
    assert persisted["gt_source"] == "empirical_search"
    np.testing.assert_array_equal(persisted["gt_reference"], gt)

    reopened = mb.mops.DTLZ2(M=2)
    assert reopened.calibrate(source_baseline=sidecar) is False
    np.testing.assert_array_equal(reopened.calibrated_gt, gt)
