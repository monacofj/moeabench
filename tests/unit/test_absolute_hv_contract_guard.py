# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract guards for MOP-owned absolute HV (ADRs 0019, 0035, 0047).

These tests must fail against the published 0.17.0 regression.
They are added as a separate file: no pre-existing regression test is edited.
"""

import numpy as np
import pytest

import moeabench as mb
from moeabench.core.run import Run


GT = np.array([
    [0.0, 1.0],
    [0.5, np.sqrt(0.75)],
    [np.sqrt(0.75), 0.5],
    [1.0, 0.0],
])
POP = np.array([[0.2, 0.9], [0.9, 0.2], [0.7, 0.7]])
OTHER = np.array([[0.1, 0.95], [0.95, 0.1], [0.55, 0.65]])


def _mop_linked_run(monkeypatch):
    mop = mb.mops.DTLZ2(M=2)
    monkeypatch.setattr(mop, "pf", lambda n_points=500: GT.copy())
    exp = mb.experiment(mop=mop)
    history = [POP.copy()]
    empty = [np.empty((len(POP), 0))]
    payload = (history, empty, POP.copy(), history, empty, empty, empty)
    return Run(payload, experiment=exp)


def _raw(front, ref):
    return float(mb.metrics.hv(front, ref=ref, scale="raw",
                               mode="exact", progress=False))


def test_absolute_hv_without_ref_uses_mop_gt(monkeypatch):
    run = _mop_linked_run(monkeypatch)
    numerator = float(mb.metrics.hv(run, scale="raw", mode="exact", progress=False))
    denominator = _raw(GT, POP)
    actual = float(mb.metrics.hv(run, scale="abs", mode="exact", progress=False))
    assert actual == pytest.approx(numerator / denominator)


def test_absolute_hv_ref_only_selects_bounds(monkeypatch):
    run = _mop_linked_run(monkeypatch)
    numerator = _raw(POP, OTHER)
    denominator = _raw(GT, OTHER)
    actual = float(mb.metrics.hv(run, ref=OTHER, scale="abs",
                                 mode="exact", progress=False))
    assert actual == pytest.approx(numerator / denominator)


def test_absolute_hv_accepts_multiple_normalization_reference_sources(monkeypatch):
    run = _mop_linked_run(monkeypatch)
    sources = [OTHER, GT]
    numerator = _raw(POP, sources)
    denominator = _raw(GT, sources)
    actual = float(mb.metrics.hv(run, ref=sources, scale="abs",
                                 mode="exact", progress=False))
    assert actual == pytest.approx(numerator / denominator)
