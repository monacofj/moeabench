# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed-input conventional metric stability, independent of MOEA trajectories.

Issue #97. Expected values are hand-derived in the versioned JSON fixture
and audited separately; never generated from the implementation under test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import moeabench as mb


_REFERENCE = json.loads(
    (Path(__file__).parent / "conventional_metric_reference_v1.json").read_text(
        encoding="utf-8"
    )
)
assert _REFERENCE["schema"] == "moeabench_conventional_metric_reference_v1"
CASES = _REFERENCE["cases"]
ZERO_CASES = _REFERENCE["zero_cases"]
ABS_TOL = _REFERENCE["provenance"]["tolerance_abs"]
INDICATORS = ("gd", "igd", "gdplus", "igdplus", "emd")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("metric", INDICATORS)
def test_frozen_distance_indicator_values(case, metric):
    front = np.asarray(case["front"], dtype=np.float64)
    reference = np.asarray(case["reference"], dtype=np.float64)
    expected = case["expected"][metric]
    value = getattr(mb.metrics, metric)(front, ref=reference, progress=False)
    assert float(value) == pytest.approx(expected, abs=ABS_TOL, rel=0)


@pytest.mark.parametrize("case", ZERO_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("metric", INDICATORS)
def test_frozen_identity_indicator_values(case, metric):
    front = np.asarray(case["front"], dtype=np.float64)
    reference = np.asarray(case["reference"], dtype=np.float64)
    value = getattr(mb.metrics, metric)(front, ref=reference, progress=False)
    assert float(value) == pytest.approx(case["expected"][metric], abs=ABS_TOL, rel=0)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("scale", ("raw", "rel"))
def test_frozen_exact_hypervolume_values(case, scale):
    front = np.asarray(case["front"], dtype=np.float64)
    reference = np.asarray(case["reference"], dtype=np.float64)
    value = mb.metrics.hv(
        front, ref=reference, mode="exact", scale=scale, progress=False
    )
    assert float(value) == pytest.approx(
        case["expected"][f"hv_{scale}"], abs=ABS_TOL, rel=0
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_frozen_absolute_hv_uses_calibrated_mop_gt(case, tmp_path):
    """Absolute HV uses MOP-owned GT, while ref sets external box geometry."""
    mop = mb.mops.DTLZ2(M=case["objectives"])
    gt = np.asarray(case["calibrated_gt"], dtype=np.float64)
    mop.calibrate(
        source_gt=gt,
        source_baseline=tmp_path / f"{case['id']}-gt.json",
    )

    class AssociatedFront(np.ndarray):
        pass

    front = np.asarray(case["front"], dtype=np.float64).view(AssociatedFront)
    front.mop = mop
    reference = np.asarray(case["reference"], dtype=np.float64)

    value = mb.metrics.hv(
        front, ref=reference, mode="exact", scale="abs", progress=False
    )
    assert float(value) == pytest.approx(
        case["expected"]["hv_abs"], abs=ABS_TOL, rel=0
    )
    # A public independent reference sets bounds, not the ground truth.
    np.testing.assert_allclose(mop.calibrated_gt, gt, atol=0, rtol=0)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_seeded_mc_hv_against_analytic_volume(case):
    """A portable analytical volume checks MC accuracy, not a host-specific hash."""
    front = np.asarray(case["front"], dtype=np.float64)
    reference = np.asarray(case["reference"], dtype=np.float64)
    args = dict(
        ref=reference, mode="fast", scale="raw",
        n_samples=80000, mc_seed=23, progress=False,
    )
    first = float(mb.metrics.hv(front, **args))
    second = float(mb.metrics.hv(front, **args))
    assert np.isfinite(first)
    assert first == second
    assert first == pytest.approx(case["expected"]["hv_raw"], abs=0.05, rel=0)
