# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise real three-indicator measurement, without recording a baseline."""
import numpy as np
from moeabench import certification as cert
import moeabench as mb


def test_explicit_case_measures_one_real_solver_run():
    case=cert._default_cases()[0].copy()
    case["moea_kwargs"]={"population":12, "generations":2}
    front=np.asarray(mb.mops.DTLZ2(M=3).pf(n_points=32), dtype=float)
    result=cert._measure(case, seed=7, front=front)
    assert result["seed"] == 7
    assert result["front_size"] > 0
    for metric in ("igdplus","gdplus","emd"):
        assert np.isfinite(result[metric])
        assert result[metric] >= 0
