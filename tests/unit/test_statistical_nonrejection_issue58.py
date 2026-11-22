# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent statistical oracles and report semantics for issue #58."""

import numpy as np
import pytest
from scipy.stats import ks_2samp, mannwhitneyu

import moeabench as mb


def test_shape_difference_can_disagree_with_rank_test():
    """Different distributions can have exactly balanced pairwise ranks."""
    a = np.array([0.0] * 20 + [10.0] * 20)
    b = np.array([4.0] * 40)

    ks_oracle = ks_2samp(a, b, alternative="two-sided")
    mw_oracle = mannwhitneyu(a, b, alternative="two-sided")
    ks = mb.stats.perf_match(a, b)
    mw = mb.stats.perf_shift(a, b)

    assert ks.statistic == pytest.approx(ks_oracle.statistic)
    assert ks.p_value == pytest.approx(ks_oracle.pvalue)
    assert mw.statistic == pytest.approx(mw_oracle.statistic)
    assert mw.p_value == pytest.approx(mw_oracle.pvalue)
    assert ks.statistic == pytest.approx(0.5)
    assert mw.statistic == pytest.approx(800.0)
    assert ks.p_value < 0.05
    assert mw.p_value > 0.05
    assert ks.decision == "divergent"
    assert mw.decision == "no-significant-difference"


@pytest.mark.parametrize("full", [False, True])
def test_ks_nonrejection_reports_no_equivalence_claim(full):
    """Preserve the decision API while qualifying its statistical meaning."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    ks = mb.stats.perf_match(a, a.copy())

    assert ks.p_value == pytest.approx(1.0)
    assert ks.decision == "match"  # Existing public API, unchanged.
    report = ks.report(show=False, full=full).lower()
    assert "did not reject" in report
    assert "does not establish equivalence" in report
