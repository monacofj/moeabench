# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import moeabench as mb


def test_perf_aliases_map_to_technical_methods():
    rng = np.random.default_rng(7)
    a = rng.normal(loc=1.0, scale=0.5, size=30)
    b = rng.normal(loc=0.0, scale=0.5, size=30)

    assert mb.stats.perf_shift(a, b).method == "mannwhitney"
    assert mb.stats.perf_match(a, b).method == "ks"
    assert mb.stats.perf_win(a, b).method == "a12"


def test_topo_aliases_map_to_technical_methods():
    a = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    b = np.array([[0.1, 0.0], [1.1, 1.0], [2.1, 2.0], [3.1, 3.0]])

    assert mb.stats.topo_match(a, b).method == "ks"
    assert mb.stats.topo_tail(a, b).method == "anderson"
    assert mb.stats.topo_shift(a, b).method == "emd"
