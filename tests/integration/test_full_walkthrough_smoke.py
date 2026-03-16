# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import moeabench as mb


def test_full_walkthrough_smoke(paired_experiments, canonical_gt):
    exp1, exp2 = paired_experiments

    assert mb.view.topology(exp1, exp2, mode="static", show=False) is not None

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
        assert left.report() is not None
        assert right.report() is not None

        shift = mb.stats.perf_shift(left, right)
        match = mb.stats.perf_match(left, right)
        win = mb.stats.perf_win(left, right)

        assert shift.report() is not None
        assert match.report() is not None
        assert win.report() is not None

        assert mb.view.history(left, right, mode="static", show=False) is not None
        assert mb.view.spread(shift, mode="static", show=False) is not None
        assert mb.view.density(match, mode="static", show=False) is not None

    topo_match = mb.stats.topo_match(exp1, exp2)
    topo_shift = mb.stats.topo_shift(exp1, exp2, threshold=0.05)
    topo_tail = mb.stats.topo_tail(exp1, exp2)
    assert topo_match is not None and topo_shift is not None and topo_tail is not None
    assert topo_match.report() is not None
    assert topo_shift.report() is not None
    assert topo_tail.report() is not None
    assert mb.view.density(mb.stats.topo_match(exp1, exp2, axes=[0]), mode="static", show=False) is not None
    assert mb.view.density(mb.stats.topo_match(exp1, exp2, space="vars", axes=[0]), mode="static", show=False) is not None

    att1 = mb.stats.attainment(exp1)
    att2 = mb.stats.attainment(exp2)
    band1_lo = mb.stats.attainment(exp1, level=0.1)
    band1_hi = mb.stats.attainment(exp1, level=0.9)
    band2_lo = mb.stats.attainment(exp2, level=0.1)
    band2_hi = mb.stats.attainment(exp2, level=0.9)
    gap = mb.stats.attainment_gap(exp1, exp2)
    assert gap.report() is not None
    assert mb.view.bands(
        att1,
        band1_lo,
        band1_hi,
        att2,
        band2_lo,
        band2_hi,
        style="fill",
        mode="static",
        show=False,
        show_gt=False,
    ) is not None
    assert mb.view.topology(att1, att2, gt=canonical_gt, mode="static", show=False) is not None
    assert mb.view.gap(gap, mode="static", show=False, show_gt=False) is not None

    ranks = mb.stats.ranks(exp1, exp2)
    strata = mb.stats.strata(exp1, exp2)
    tiers = mb.stats.tiers(exp1, exp2)
    assert ranks.report() is not None
    assert strata.report() is not None
    assert tiers.report() is not None
    assert mb.view.ranks(ranks, mode="static", show=False) is not None
    assert mb.view.strata(strata, mode="static", show=False) is not None
    assert mb.view.tiers(tiers, mode="static", show=False) is not None

    diag1 = mb.clinic.audit(exp1)
    diag2 = mb.clinic.audit(exp2)
    close1 = mb.clinic.closeness(exp1, ref=canonical_gt)
    close2 = mb.clinic.closeness(exp2, ref=canonical_gt)

    assert diag1.report(full=True) is not None
    assert diag2.report(full=True) is not None
    assert close1.report() is not None
    assert close2.report() is not None

    assert mb.view.radar(diag1, diag2, mode="static", show=False) is not None
    assert mb.view.ecdf(close1, mode="static", show=False) is not None
    assert mb.view.ecdf(close2, mode="static", show=False) is not None
    assert mb.view.density(close1, mode="static", show=False) is not None
    assert mb.view.density(close2, mode="static", show=False) is not None
    assert mb.view.history(close1, mode="static", show=False) is not None
    assert mb.view.history(close2, mode="static", show=False) is not None
