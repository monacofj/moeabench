# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collected, lightweight guards for the three public MOP GT source protocols.

The former verify_gt_protocols.py is not collected by pytest. These tests
exercise its CSV and empirical-search paths without writing in the repo or
running an expensive optimization. Analytical GT and persistence are already
covered by test_gt_calibration_independent.py.
"""

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import moeabench as mb


def test_gt_from_csv_is_persisted_and_reloadable(tmp_path):
    source = tmp_path / "external-front.csv"
    sidecar = tmp_path / "DTLZ2-gt.json"
    gt = np.array([[0.0, 1.0], [0.25, 0.75], [1.0, 0.0]])
    np.savetxt(source, gt, delimiter=",", fmt="%.17g")

    mop = mb.mops.DTLZ2(M=2)
    assert mop.calibrate(source_gt=source, source_baseline=sidecar) is True
    np.testing.assert_array_equal(mop.calibrated_gt, gt)

    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    assert saved["schema"] == "mop_gt_v1"
    assert saved["gt_source"] == "external"
    np.testing.assert_array_equal(saved["gt_reference"], gt)

    reloaded = mb.mops.DTLZ2(M=2)
    assert reloaded.calibrate(source_baseline=sidecar) is False
    np.testing.assert_array_equal(reloaded.calibrated_gt, gt)


def test_source_search_uses_empirical_nondominated_objectives(monkeypatch, tmp_path):
    """Test the calibration adapter without making a unit test run a MOEA."""
    experiment_module = importlib.import_module("moeabench.core.experiment")
    gt = np.array([[0.1, 0.9], [0.4, 0.6], [0.9, 0.1]])
    observed = {}

    class SearchExperiment:
        def __init__(self, mop, moea):
            observed["mop"] = mop
            observed["moea"] = moea

        def run(self):
            observed["run"] = True

        def non_dominated(self):
            observed["non_dominated"] = True
            return SimpleNamespace(objectives=gt.copy())

    monkeypatch.setattr(experiment_module, "experiment", SearchExperiment)
    mop = mb.mops.DTLZ2(M=2)
    search = object()
    sidecar = tmp_path / "empirical.json"

    assert mop.calibrate(source_search=search, source_baseline=sidecar) is True
    assert observed == {
        "mop": mop, "moea": search, "run": True, "non_dominated": True,
    }
    np.testing.assert_array_equal(mop.calibrated_gt, gt)
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    assert saved["gt_source"] == "empirical_search"
    np.testing.assert_array_equal(saved["gt_reference"], gt)


def test_failed_sidecar_write_preserves_existing_calibrated_gt(tmp_path):
    mop = mb.mops.DTLZ2(M=2)
    original = np.array([[0.0, 1.0], [1.0, 0.0]])
    replacement = np.array([[0.2, 0.8], [0.8, 0.2]])

    assert mop.calibrate(
        source_baseline=tmp_path / "original.json", source_gt=original
    ) is True
    with pytest.raises(OSError):
        mop.calibrate(
            source_baseline=tmp_path / "missing-directory" / "new.json",
            source_gt=replacement,
            force=True,
        )

    np.testing.assert_array_equal(mop.calibrated_gt, original)


def test_invalid_sidecar_cannot_be_silently_accepted(tmp_path):
    mop = mb.mops.DTLZ2(M=2)
    sidecar = tmp_path / "invalid.json"
    sidecar.write_text(
        json.dumps({
            "schema": "mop_gt_v1",
            "problem_id": mop.name,
            "mop_dimension": 3,
            "gt_reference": [[0.0, 1.0, 0.0]],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different objective count"):
        mop.calibrate(source_baseline=sidecar)
