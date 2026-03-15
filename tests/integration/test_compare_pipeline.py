# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import moeabench as mb


def test_compare_pipeline(paired_experiments):
    exp1, exp2 = paired_experiments

    hv1 = mb.metrics.hv(exp1, scale="abs")
    hv2 = mb.metrics.hv(exp2, scale="abs")

    perf_shift = mb.stats.perf_shift(hv1, hv2)
    perf_match = mb.stats.perf_match(hv1, hv2)
    perf_win = mb.stats.perf_win(hv1, hv2)

    assert perf_shift.method == "mannwhitney"
    assert perf_match.method == "ks"
    assert perf_win.method == "a12"
    assert perf_win.effect_size is not None

    topo_match = mb.stats.topo_match(exp1, exp2)
    topo_shift = mb.stats.topo_shift(exp1, exp2, threshold=0.05)
    topo_tail = mb.stats.topo_tail(exp1, exp2)
    topo_match_obj = mb.stats.topo_match(exp1, exp2, axes=[0])
    topo_match_var = mb.stats.topo_match(exp1, exp2, space="vars", axes=[0])

    assert topo_match.method == "ks"
    assert topo_shift.method == "emd"
    assert topo_tail.method == "anderson"
    assert topo_match_obj.axes == [0]
    assert topo_match_var.space == "vars"
    assert mb.view.density(topo_match_obj, mode="static", show=False) is not None
    assert mb.view.density(topo_match_var, mode="static", show=False) is not None
