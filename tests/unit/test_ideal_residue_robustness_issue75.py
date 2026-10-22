# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Issue #75: test sensitivity without blessing the first analytical PF prefix."""

import numpy as np
import pytest

from calibration.scripts.study_ideal_residue_robustness import (
    _fixed_q, _region_hist, compare_subsets,
)


def _data():
    x = np.linspace(0, 1, 121)
    gt = np.column_stack((x, 1-x, np.zeros_like(x)))
    xx = np.linspace(0, 1, 4608)
    # Every analytical point is off the stored 3D GT by a positive,
    # deliberately heterogeneous distance; this tests variation rather
    # than relying on the happy case of a perfectly constant epsilon.
    zz = 0.005 + 0.02 * xx
    pool = np.column_stack((xx, 1-xx, zz))
    blur = np.linspace(0.05, 2.0, 200)
    return gt, pool, {"20": blur, "100": blur * 1.05}


def test_robustness_exposes_prefix_bias_and_preserves_disjoint_replicates():
    gt, pool, blur = _data()
    # The ordered prefix is concentrated in the first quarter of f1 and
    # gives a different epsilon from the full-domain random partitions.
    prefix = pool[:1024]
    report = compare_subsets(
        gt, pool, prefix, blur,
        sample_sizes=(256, 512, 1024), repetitions=4,
        k_values=(20, 100), seed=75,
    )
    assert report["candidate_pool_count"] == 4608
    assert len(report["sample_sizes"]["1024"]["replicates"]) == 4
    assert report["baseline_first_1024"]["regions_f1"][0] > 800
    assert report["sample_sizes"]["1024"]["summary"]["100"]["epsilon"]["min"] > 0
    old = report["baseline_first_1024"]["100"]["epsilon"]
    new = report["sample_sizes"]["1024"]["summary"]["100"]["epsilon"]["median"]
    assert old < new
    assert report["sample_sizes"]["1024"]["summary"]["100"]["legacy_prefix_outside_replicate_range"]
    for size in report["sample_sizes"].values():
        for row in size["replicates"]:
            assert sum(row["region_f1_counts"]) == row["n"]
            assert set(row["k"]) == {"20", "100"}
            for value in row["k"].values():
                assert 0 <= value["q"]["0.35"] <= 1
                assert 0 <= value["q"]["0.65"] <= 1

    # Same seed and fixed candidates must produce bitwise identical summary.
    assert report == compare_subsets(
        gt, pool, prefix, blur,
        sample_sizes=(256, 512, 1024), repetitions=4,
        k_values=(20, 100), seed=75,
    )


def test_closeness_q_sensitivity_is_measured_against_equally_corrected_ecdf():
    raw = np.linspace(0.01, 2.0, 200)
    for eps in (0, 0.05, 0.30):
        for factor in (0.35, 0.65):
            q = _fixed_q(raw * factor, raw, eps, k=100)
            assert np.isfinite(q) and 0 <= q <= 1
    assert _fixed_q(raw, raw, 0.05) == pytest.approx(0.0)


def test_bad_candidate_pool_does_not_silently_produce_scientific_result():
    gt, pool, blur = _data()
    with pytest.raises(ValueError, match="Insufficient"):
        compare_subsets(gt, pool[:1024], pool[:1024], blur,
                        sample_sizes=(1024,), repetitions=4)
    bad_pool = pool.copy()
    bad_pool[0] = gt[0]
    with pytest.raises(ValueError, match="overlaps"):
        compare_subsets(gt, bad_pool, pool[:1024], blur,
                        sample_sizes=(256,), repetitions=4)


def test_missing_k_matched_ecdf_fails_closed():
    gt, pool, blur = _data()
    with pytest.raises(ValueError, match="K=100"):
        compare_subsets(
            gt, pool, pool[:1024], {"20": blur["20"]},
            sample_sizes=(256,), repetitions=4,
            k_values=(20, 100),
        )


def test_region_occupancy_exposes_one_axis_clustering():
    gt, pool, _ = _data()
    first = _region_hist(pool[:600], gt)
    assert sum(first) == 600
    assert first[0] > 500
    distributed = _region_hist(pool, gt)
    assert all(n > 0 for n in distributed)
