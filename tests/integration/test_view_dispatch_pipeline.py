# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import moeabench as mb


def test_view_dispatch_pipeline(paired_experiments, canonical_gt):
    exp1, exp2 = paired_experiments

    hv1 = mb.metrics.hv(exp1, scale="abs")
    hv2 = mb.metrics.hv(exp2, scale="abs")
    topo_match = mb.stats.topo_match(exp1, exp2, axes=[0])
    close1 = mb.clinic.closeness(exp1, ref=canonical_gt)

    perf_hist = mb.view.history(hv1, hv2, show=False)
    clinic_hist = mb.view.history(close1, show=False)
    topo_density = mb.view.density(topo_match, show=False)
    clinic_density = mb.view.density(close1, show=False)

    assert isinstance(perf_hist, tuple) and perf_hist[0] is not None
    assert clinic_hist is not None
    assert topo_density is not None
    assert clinic_density is not None
