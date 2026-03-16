# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Deep statistical stability suite for algorithmic performance certification."""

from __future__ import annotations

import os
import zlib

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

import moeabench as mb


BASELINE_FILE = "tests/baselines_v0.8.0.csv"
RUN_DEEP = os.getenv("MOEABENCH_RUN_DEEP", "").strip().lower() in {"1", "true", "yes"}


def get_deep_configs():
    if not os.path.exists(BASELINE_FILE):
        return []

    df = pd.read_csv(BASELINE_FILE)
    # Legacy deep baselines were recorded under the old "standard" intensity label.
    std_df = df[df["Intensity"] == "standard"]
    return [
        (row["MOP"], row["Algorithm"], row["IGD_mean"], row["IGD_std"], row["Pop"], row["Gen"])
        for _, row in std_df.iterrows()
    ]


CONFIGS = get_deep_configs()


def _build_problem(mop_name: str):
    mop_cls = getattr(mb.mops, mop_name)
    try:
        return mop_cls(M=3)
    except TypeError:
        return mop_cls(M=3, D=2)


@pytest.mark.slow
@pytest.mark.skipif(not RUN_DEEP, reason="Deep tier disabled (set MOEABENCH_RUN_DEEP=1 to enable).")
@pytest.mark.skipif(not os.path.exists(BASELINE_FILE), reason="Deep baseline CSV not found.")
@pytest.mark.parametrize("mop_name, alg_name, base_igd_mean, base_igd_std, pop, gen", CONFIGS)
def test_deep_statistical_quality(mop_name, alg_name, base_igd_mean, base_igd_std, pop, gen):
    n_repeats = 30
    mop = _build_problem(mop_name)
    ref = mop.pf(n_points=2000) if hasattr(mop, "pf") else mb.experiment(mop=mop).optimal_front(n_points=2000)

    current_igd_values = []
    for i in range(n_repeats):
        run_seed = zlib.crc32(f"deep_{mop_name}_{alg_name}_{i}".encode()) & 0xFFFFFFFF
        alg = getattr(mb.moeas, alg_name)(seed=run_seed, population=int(pop), generations=int(gen))
        exp = mb.experiment(mop=mop)
        exp.moea = alg
        exp.run(repeat=1, silent=True)
        igd = float(mb.metrics.igd(np.asarray(exp.pop()), ref=ref))
        current_igd_values.append(igd)

    current_mean = float(np.mean(current_igd_values))
    t_stat, p_val = stats.ttest_ind_from_stats(
        mean1=current_mean,
        std1=np.std(current_igd_values, ddof=1),
        nobs1=n_repeats,
        mean2=base_igd_mean,
        std2=base_igd_std,
        nobs2=30,
        alternative="greater",
    )
    if p_val < 0.05:
        pooled_std = np.sqrt((np.std(current_igd_values) ** 2 + base_igd_std ** 2) / 2)
        cohens_d = (current_mean - base_igd_mean) / pooled_std
        if cohens_d > 0.5:
            pytest.fail(f"Performance regression detected: p={p_val:.4f}, d={cohens_d:.2f}")
