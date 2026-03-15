# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import moeabench as mb


def test_metrics_pipeline(paired_experiments, canonical_gt):
    exp1, exp2 = paired_experiments

    metric_pairs = [
        (mb.metrics.hv(exp1, scale="abs"), mb.metrics.hv(exp2, scale="abs")),
        (mb.metrics.gd(exp1, ref=canonical_gt), mb.metrics.gd(exp2, ref=canonical_gt)),
        (mb.metrics.gdplus(exp1, ref=canonical_gt), mb.metrics.gdplus(exp2, ref=canonical_gt)),
        (mb.metrics.igd(exp1, ref=canonical_gt), mb.metrics.igd(exp2, ref=canonical_gt)),
        (mb.metrics.igdplus(exp1, ref=canonical_gt), mb.metrics.igdplus(exp2, ref=canonical_gt)),
        (mb.metrics.emd(exp1, ref=canonical_gt), mb.metrics.emd(exp2, ref=canonical_gt)),
        (mb.metrics.front_ratio(exp1), mb.metrics.front_ratio(exp2)),
    ]

    for left, right in metric_pairs:
        assert hasattr(left, "report")
        assert hasattr(right, "report")
        assert left.values.shape[1] == len(exp1.runs)
        assert right.values.shape[1] == len(exp2.runs)
        assert mb.view.history(left, right, mode="static", show=False)[0] is not None

        shift = mb.stats.perf_shift(left, right)
        match = mb.stats.perf_match(left, right)

        assert shift.method == "mannwhitney"
        assert match.method == "ks"
        assert mb.view.spread(shift, mode="static", show=False) is not None
        assert mb.view.density(match, mode="static", show=False) is not None
