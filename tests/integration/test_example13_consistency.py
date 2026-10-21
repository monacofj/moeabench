# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""The example's quality report and clinical plot must describe the same run."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import moeabench as mb


@pytest.mark.parametrize("use_calibration_bounds", [False, True])
def test_example13_audit_and_topology_share_run_and_reference(monkeypatch, use_calibration_bounds):
    # An interface-level regression: avoid a 280-generation optimization run
    # while checking that every public API call receives the intended object.
    gt = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0]])
    exp = SimpleNamespace(run=Mock(), optimal_front=Mock(return_value=gt))
    mop, moea = object(), object()
    report = Mock()
    normalization = (
        {"ideal": np.array([1.0, 2.0, 3.0]), "nadir": np.array([3.0, 4.0, 7.0])}
        if use_calibration_bounds else None
    )
    audit = Mock(return_value=SimpleNamespace(
        quality=SimpleNamespace(report=report),
        diagnostic_context={"GT": gt, "normalization": normalization},
    ))
    plot = Mock()
    constructor = Mock(return_value=exp)
    mop_factory = Mock(return_value=mop)
    moea_factory = Mock(return_value=moea)

    monkeypatch.setattr(mb, "experiment", constructor)
    monkeypatch.setattr(mb.system, "version", Mock())
    monkeypatch.setattr(mb.mops, "DTLZ1", mop_factory)
    monkeypatch.setattr(mb.moeas, "NSGA2", moea_factory)
    monkeypatch.setattr(mb.clinic, "audit", audit)
    monkeypatch.setattr(mb.view, "topology", plot)
    # The example uses mb_path when launched directly from the examples folder.
    monkeypatch.setitem(sys.modules, "mb_path", ModuleType("mb_path"))

    script = Path(__file__).resolve().parents[2] / "examples" / "example_13.py"
    runpy.run_path(str(script), run_name="__main__")

    constructor.assert_called_once_with()
    mop_factory.assert_called_once_with(M=3)
    moea_factory.assert_called_once_with(population=100, generations=280)
    assert exp.mop is mop
    assert exp.moea is moea
    exp.run.assert_called_once_with(seed=1)
    exp.optimal_front.assert_not_called()
    audit.assert_called_once_with(exp)
    report.assert_called_once_with(full=True)
    expected_plot_gt = gt if normalization is None else (
        gt * (normalization["nadir"] - normalization["ideal"]) + normalization["ideal"]
    )
    plot.assert_called_once()
    (plot_args, plot_kwargs) = plot.call_args
    assert plot_args == (exp,)
    assert plot_kwargs["markers"] is True
    np.testing.assert_array_equal(plot_kwargs["gt"], expected_plot_gt)
