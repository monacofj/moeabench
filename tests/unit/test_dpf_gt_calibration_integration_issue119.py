# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Corrected native DPF3/DPF5 PS/PF and explicit GT calibration integration.

Calibration is called against temporary standalone sidecars; the test never
writes to the repository, changes default budgets or touches FAIR payloads.
"""
import json

import numpy as np
import pytest

from moeabench.mops.DPF3 import DPF3
from moeabench.mops.DPF5 import DPF5


@pytest.mark.parametrize("cls,m,d,n", [
    (DPF3,5,2,128),
    (DPF3,6,3,128),
    (DPF5,3,2,128),
    (DPF5,5,2,128),
    (DPF5,6,4,128),
])
def test_native_corrected_gt_calibration_sidecar_reload_and_defensive_copy(cls,m,d,n,tmp_path):
    path=tmp_path / "inaugural-dpf-gt.json"
    mop=cls(M=m,D=d)
    expected=mop.pf(n_points=n)

    # The first explicit request generates/persists a valid analytical GT.
    assert mop.calibrate(source_baseline=str(path),n_points=n) is True
    assert path.is_file()
    np.testing.assert_array_equal(mop.calibrated_gt,expected)

    sidecar=json.loads(path.read_text(encoding="utf-8"))
    assert sidecar["schema"]=="mop_gt_v1"
    assert sidecar["problem_id"]==mop.name
    assert sidecar["mop_dimension"]==m
    assert sidecar["gt_source"]=="analytical"
    assert len(sidecar["gt_reference"])==n
    assert len(np.unique(sidecar["gt_reference"],axis=0))==n
    np.testing.assert_array_equal(np.asarray(sidecar["gt_reference"],dtype=float),expected)

    # Caller mutations must not change the calibrated reference.
    exported=mop.calibrated_gt
    exported[0,0]+=17
    np.testing.assert_array_equal(mop.calibrated_gt,expected)

    # A fresh MOP reloads the same data without regenerating or changing it.
    replay=cls(M=m,D=d)
    assert replay.calibrate(source_baseline=str(path)) is False
    np.testing.assert_array_equal(replay.calibrated_gt,expected)

    # Rerun only if explicitly forced; same equations, deterministic result.
    assert replay.calibrate(source_baseline=str(path),n_points=n,force=True) is True
    np.testing.assert_array_equal(replay.calibrated_gt,expected)
