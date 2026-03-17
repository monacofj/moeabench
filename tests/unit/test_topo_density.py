# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

import moeabench as mb


def test_topology_places_gt_trace_before_experiment_trace():
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ2(M=3)
    exp.moea = mb.moeas.NSGA2deap(population=20, generations=3, seed=7)
    exp.name = "NSGA-II"
    exp.run(repeat=1, silent=True)

    plot = mb.view.topology(exp, show=False, mode="interactive")
    names = [trace.name for trace in plot.figure.data]

    assert names[0] == "True Front (GT)"
    assert "NSGA-II" in names[1]


def test_topo_density_uses_shared_axis_domain_for_kde_curves():
    a = np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.4], [0.3, 0.6]])
    b = np.array([[10.0, 0.0], [10.1, 0.2], [10.2, 0.4], [10.3, 0.6]])

    fig = mb.view.density(a, b, axes=[0], mode="static", show=False)
    ax = fig.axes[0]

    assert len(ax.lines) >= 2
    x0 = ax.lines[0].get_xdata()
    x1 = ax.lines[1].get_xdata()

    assert np.array_equal(x0, x1)
    assert np.isclose(x0[0], 0.0)
    assert np.isclose(x0[-1], 10.3)


def test_perf_density_uses_shared_axis_domain_for_kde_curves():
    a = np.array([0.0, 0.1, 0.2, 0.3])
    b = np.array([10.0, 10.1, 10.2, 10.3])

    fig = mb.view.density(a, b)
    ax = fig.axes[0]

    assert len(ax.lines) >= 2
    x0 = ax.lines[0].get_xdata()
    x1 = ax.lines[1].get_xdata()

    assert np.array_equal(x0, x1)
    assert np.isclose(x0[0], 0.0)
    assert np.isclose(x0[-1], 10.3)
