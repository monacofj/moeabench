# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import moeabench as mb


def test_attainment_pipeline(paired_experiments, canonical_gt):
    exp1, exp2 = paired_experiments

    att1 = mb.stats.attainment(exp1)
    att2 = mb.stats.attainment(exp2)
    band1_lo = mb.stats.attainment(exp1, level=0.1)
    band1_hi = mb.stats.attainment(exp1, level=0.9)
    band2_lo = mb.stats.attainment(exp2, level=0.1)
    band2_hi = mb.stats.attainment(exp2, level=0.9)
    gap = mb.stats.attainment_gap(exp1, exp2)

    assert att1.level == 0.5
    assert att2.level == 0.5
    assert hasattr(gap, "report")

    assert mb.view.bands(
        att1, band1_lo, band1_hi, att2, band2_lo, band2_hi,
        mode="static", show=False, show_gt=False
    ) is not None
    assert mb.view.topology(att1, att2, gt=canonical_gt, mode="static", show=False) is not None
    assert mb.view.gap(gap, mode="static", show=False, show_gt=False) is not None
