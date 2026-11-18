# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent MOP GT / H_abs integration guards (ADRs 0019, 0035, 0047)."""

import numpy as np
import pytest
import moeabench as mb
from moeabench.core.run import Run


GT = np.array([[0.0, 1.0], [0.5, np.sqrt(0.75)],
               [np.sqrt(0.75), 0.5], [1.0, 0.0]])
POP = np.array([[0.2, 0.9], [0.9, 0.2], [0.7, 0.7]])
REF = np.array([[0.1, 0.95], [0.95, 0.1], [0.55, 0.65]])


def _run(mop):
    exp = mb.experiment(mop=mop)
    history = [POP.copy()]
    empty = [np.empty((len(POP), 0))]
    return Run((history, empty, POP.copy(), history, empty, empty, empty),
               experiment=exp)


def _raw(front, ref=None):
    return float(mb.metrics.hv(front, ref=ref, scale="raw",
                               mode="exact", progress=False))


@pytest.mark.parametrize("ref", [None, REF, [REF, GT]])
def test_absolute_hv_uses_calibrated_mop_gt_not_geometry_ref(tmp_path, ref):
    mop = mb.mops.DTLZ2(M=2)
    mop.calibrate(source_baseline=tmp_path / "mop-gt.json", source_gt=GT)
    run = _run(mop)
    actual = mb.metrics.hv(run, ref=ref, scale="abs",
                           mode="exact", progress=False)
    expected_num = _raw(POP, ref=ref)
    expected_den = _raw(GT, ref=POP if ref is None else ref)
    assert float(actual) == pytest.approx(expected_num / expected_den)
    assert actual.diagnostics["hv_abs_gt_points"] == len(GT)
    assert actual.diagnostics["hv_abs_denominator"] == pytest.approx(expected_den)


def test_absolute_hv_rejects_uncalibrated_mop(tmp_path):
    run = _run(mb.mops.DTLZ2(M=2))
    with pytest.raises(ValueError, match="Call mop.calibrate|call mop.calibrate"):
        mb.metrics.hv(run, scale="abs", mode="exact", progress=False)


def test_absolute_hv_rejects_unowned_front_even_with_geometry_ref():
    with pytest.raises(ValueError, match="MOP-associated"):
        mb.metrics.hv(POP, ref=GT, scale="abs", mode="exact", progress=False)
