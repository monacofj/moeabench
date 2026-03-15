# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import moeabench as mb


def test_full_walkthrough_smoke(paired_experiments, canonical_gt):
    exp1, exp2 = paired_experiments

    mb.view.topology(exp1, exp2, mode="static", show=False)

    hv1 = mb.metrics.hv(exp1, scale="abs")
    hv2 = mb.metrics.hv(exp2, scale="abs")
    gd1 = mb.metrics.gd(exp1, ref=canonical_gt)
    gd2 = mb.metrics.gd(exp2, ref=canonical_gt)
    igd1 = mb.metrics.igd(exp1, ref=canonical_gt)
    igd2 = mb.metrics.igd(exp2, ref=canonical_gt)
    emd1 = mb.metrics.emd(exp1, ref=canonical_gt)
    emd2 = mb.metrics.emd(exp2, ref=canonical_gt)
    fsize1 = mb.metrics.front_ratio(exp1)
    fsize2 = mb.metrics.front_ratio(exp2)

    for left, right in ((hv1, hv2), (gd1, gd2), (igd1, igd2), (emd1, emd2), (fsize1, fsize2)):
        mb.stats.perf_shift(left, right)
        mb.stats.perf_match(left, right)
        mb.stats.perf_win(left, right)
        mb.view.history(left, right, mode="static", show=False)

    topo_match = mb.stats.topo_match(exp1, exp2)
    topo_shift = mb.stats.topo_shift(exp1, exp2, threshold=0.05)
    topo_tail = mb.stats.topo_tail(exp1, exp2)
    assert topo_match is not None and topo_shift is not None and topo_tail is not None
    mb.view.density(mb.stats.topo_match(exp1, exp2, axes=[0]), mode="static", show=False)
    mb.view.density(mb.stats.topo_match(exp1, exp2, space="vars", axes=[0]), mode="static", show=False)

    att1 = mb.stats.attainment(exp1)
    att2 = mb.stats.attainment(exp2)
    gap = mb.stats.attainment_gap(exp1, exp2)
    mb.view.bands(att1, mb.stats.attainment(exp1, level=0.1), mb.stats.attainment(exp1, level=0.9),
                  att2, mb.stats.attainment(exp2, level=0.1), mb.stats.attainment(exp2, level=0.9),
                  mode="static", show=False, show_gt=False)
    mb.view.topology(att1, att2, gt=canonical_gt, mode="static", show=False)
    mb.view.gap(gap, mode="static", show=False, show_gt=False)

    ranks = mb.stats.ranks(exp1, exp2)
    strata = mb.stats.strata(exp1, exp2)
    tiers = mb.stats.tiers(exp1, exp2)
    mb.view.ranks(ranks, mode="static", show=False)
    mb.view.strata(strata, mode="static", show=False)
    mb.view.tiers(tiers, mode="static", show=False)

    diag1 = mb.clinic.audit(exp1)
    diag2 = mb.clinic.audit(exp2)
    close1 = mb.clinic.closeness(exp1, ref=canonical_gt)
    close2 = mb.clinic.closeness(exp2, ref=canonical_gt)

    mb.view.radar(diag1, diag2, mode="static", show=False)
    mb.view.ecdf(close1, mode="static", show=False)
    mb.view.ecdf(close2, mode="static", show=False)
    mb.view.density(close1, mode="static", show=False)
    mb.view.density(close2, mode="static", show=False)
    mb.view.history(close1, mode="static", show=False)
    mb.view.history(close2, mode="static", show=False)
