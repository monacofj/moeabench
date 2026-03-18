# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np

from moeabench.metrics.evaluator import MetricMatrix
from moeabench.view.perf import perf_history
from moeabench.view.perf import _plot_metric_matrices


class _DummyMetricMatrix:
    def __init__(self, values, metric_name="GD", source_name="Exp"):
        self.values = np.asarray(values)
        self.metric_name = metric_name
        self.source_name = source_name


def test_plot_metric_matrices_returns_none_after_notebook_show(monkeypatch):
    shown = []

    import IPython
    import plotly.graph_objects as go

    monkeypatch.setattr(IPython, "get_ipython", lambda: object())
    monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(self))

    mat = _DummyMetricMatrix([[0.3], [0.2], [0.1]])
    res = _plot_metric_matrices([mat], mode="interactive", show=True)

    assert res is None
    assert len(shown) == 1


def test_plot_metric_matrices_keeps_figure_when_show_is_false():
    mat = _DummyMetricMatrix([[0.3], [0.2], [0.1]])

    res = _plot_metric_matrices([mat], mode="interactive", show=False)

    assert res is not None


def test_perf_history_slices_precomputed_metric_matrix_by_generation():
    mat = MetricMatrix([[0.3], [0.2], [0.1]], metric_name="HV", source_name="Exp")

    fig, ax = perf_history(mat, gens=2, mode="static", show=False)

    line = ax.get_lines()[0]
    assert list(line.get_xdata()) == [1, 2]
    assert np.allclose(line.get_ydata(), [0.3, 0.2])


def test_metric_report_markdown_avoids_github_admonition_tokens():
    mat = MetricMatrix([[0.3], [0.2], [0.1]], metric_name="Hypervolume (Raw)", source_name="Exp")

    report = mat.report(show=False, markdown=True)

    assert "[!NOTE]" not in report
    assert "Physical Objective Space" in report
