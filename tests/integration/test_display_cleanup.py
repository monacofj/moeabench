# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import matplotlib.pyplot as plt
import moeabench as mb


def test_topology_show_closes_internal_figures_and_explains_headless_mode(paired_experiments):
    exp1, exp2 = paired_experiments
    plt.close("all")

    plot = mb.view.topology(exp1, exp2, mode="static", show=True)
    assert plot is not None
    assert plt.get_fignums() == []
