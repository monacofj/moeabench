# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New read-only GT study tests; existing regressions remain unchanged."""
import numpy as np
import pytest

from scripts.audit.gt_complete_cost_reference_issue119 import measure, summarize


@pytest.mark.parametrize("name", ["DTLZ1", "DTLZ2"])
def test_gt_methods_hold_identical_front_across_reference_sizes(name):
    a = measure(name, 3, 12, pool_size=48,
                reference_sizes=(64, 128, 256), seed=123, iterations=3)
    b = measure(name, 3, 12, pool_size=48,
                reference_sizes=(64, 128, 256), seed=123, iterations=3)
    assert a["reference_diagnostics"] == b["reference_diagnostics"]
    assert a["generated_gt_is_fixed_across_reference_sizes"]
    assert set(a["full_generation_seconds"]) == {
        "incumbent", "anchored", "fps", "anchored_mc_cvt", "fps_mc_cvt"}
    assert all(np.isfinite(t) and t >= 0
               for t in a["full_generation_seconds"].values())
    assert max(a["inverse_error"].values()) < 1e-10
    for size in ("64", "128", "256"):
        for stats in a["reference_diagnostics"][size].values():
            assert stats["n"] == 12
            assert np.isfinite(stats["reference_to_front_mean"])
    summary = summarize([a, b])
    assert summary[0]["paired_seeds"] == 2
    assert set(summary[0]["methods"]["fps"]["reference_sizes"]) == {
        "64", "128", "256"}


def test_reject_invalid_study_inputs():
    with pytest.raises(ValueError):
        measure("DTLZ2", 5, 3, pool_size=24,
                reference_sizes=(32,), seed=12)
    with pytest.raises(ValueError, match="Reference sizes"):
        measure("DTLZ1", 3, 8, pool_size=24,
                reference_sizes=(1,), seed=12)
