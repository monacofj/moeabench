# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additional GT-only calibration guards, independent of FAIR regressions."""

import json
import numpy as np
import pytest

from moeabench.mops.base_mop import BaseMop


class _TestMop(BaseMop):
    def __init__(self, name="DTLZ_TEST"):
        super().__init__(name=name, M=2, N=2)
        self.requests = []

    def evaluation(self, X, n_ieq_constr=0):
        return {"F": np.asarray(X, dtype=float)}

    def pf(self, n_points=100):
        self.requests.append(n_points)
        x = np.linspace(0, 1, min(n_points, 5))
        return np.column_stack((x, 1 - x))


def test_gt_calibration_is_independent_and_reloadable(tmp_path):
    mop = _TestMop()
    path = tmp_path / "gt.json"
    assert mop.calibrate(source_baseline=path, n_points=7) is True
    original = mop.calibrated_gt
    assert original.shape == (5, 2)
    assert mop.requests == [7]
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema"] == "mop_gt_v1"
    assert data["problem_id"] == "DTLZ_TEST"
    assert "gt_reference" in data
    assert "problems" not in data

    original[0, 0] = 1234
    assert mop.calibrated_gt[0, 0] != 1234
    assert mop.calibrate(source_baseline=path) is False
    assert mop.requests == [7]
    assert np.array_equal(mop.calibrated_gt, np.asarray(data["gt_reference"]))


def test_source_gt_and_legacy_sidecar_read(tmp_path):
    mop = _TestMop()
    path = tmp_path / "gt.json"
    truth = np.array([[0.1, 0.9], [0.9, 0.1]])
    assert mop.calibrate(source_baseline=path, source_gt=truth) is True
    assert np.array_equal(mop.calibrated_gt, truth)
    legacy = {
        "schema": "baselines_v4_ecdf",
        "problem_id": mop.name,
        "mop_dimension": mop.M,
        "gt_reference": truth.tolist(),
        "problems": {mop.name: {"50": {"closeness": "FAIR_ONLY"}}},
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert mop.calibrate(source_baseline=path) is False
    assert np.array_equal(mop.calibrated_gt, truth)


def test_gt_calibration_rejects_mismatch_and_nonfinite(tmp_path):
    mop = _TestMop()
    path = tmp_path / "gt.json"
    wrong = {"problem_id": "OTHER", "mop_dimension": 2, "gt_reference": [[0.0, 1.0]]}
    path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="different MOP"):
        mop.calibrate(source_baseline=path)
    with pytest.raises(ValueError, match="finite"):
        mop.calibrate(source_baseline=path, force=True, source_gt=[[np.nan, 0.0]])


def test_default_gt_density_respects_dtlz_dpf_reference_policy(tmp_path):
    d = _TestMop("DTLZ_TEST")
    p = _TestMop("DPF_TEST")
    assert d.calibrate(source_baseline=tmp_path / "d.json") is True
    assert p.calibrate(source_baseline=tmp_path / "p.json") is True
    assert d.requests == [2000]
    assert p.requests == [10000]


def test_fair_only_calibration_options_do_not_silently_succeed(tmp_path):
    mop = _TestMop()
    with pytest.raises(NotImplementedError, match="FAIR/ECDF"):
        mop.calibrate(source_baseline=tmp_path / "gt.json", k_values=[20, 50])


def test_uncalibrated_mop_reports_missing_gt():
    mop = _TestMop()
    with pytest.raises(ValueError, match="Call mop.calibrate"):
        _ = mop.calibrated_gt
