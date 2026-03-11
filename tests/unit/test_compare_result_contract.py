# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import moeabench as mb


def test_perf_compare_contract_unified():
    # Simple two-sample vectors
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.5, 2.5, 3.5, 4.5])

    for method in ("shift", "match", "win"):
        res = mb.stats.perf_compare(a, b, method=method)
        assert res.method == method
        assert hasattr(res, "report")
        assert res.report(show=False, full=False).strip()


def test_topo_compare_contract_unified():
    a = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    b = np.array([[0.1, 0.0], [1.1, 1.0], [2.1, 2.0]])

    for method in ("match", "emd", "anderson"):
        res = mb.stats.topo_compare(a, b, method=method, space="objs")
        assert res.method == method
        assert hasattr(res, "report")
        assert res.report(show=False, full=False).strip()
