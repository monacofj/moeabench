# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import moeabench as mb


def test_structure_pipeline(paired_experiments):
    exp1, exp2 = paired_experiments

    ranks = mb.stats.ranks(exp1, exp2)
    strata = mb.stats.strata(exp1, exp2)
    tiers = mb.stats.tiers(exp1, exp2)

    for result in (ranks, strata, tiers):
        assert hasattr(result, "report")

    assert mb.view.ranks(ranks, mode="static", show=False) is not None
    assert mb.view.strata(strata, mode="static", show=False) is not None
    assert mb.view.tiers(tiers, mode="static", show=False) is not None
