# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np

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
