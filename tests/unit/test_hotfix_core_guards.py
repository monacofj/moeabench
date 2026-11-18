# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additional contract guards. Does not replace or weaken baseline regression tests."""

import numpy as np
import pytest
import moeabench as mb


def test_removed_fair_topology_markers_fail_explicitly():
    front = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(NotImplementedError, match="clinical quality markers"):
        mb.view.topology(front, markers=True, show=False)


def test_removed_fair_density_domain_fails_explicitly():
    front = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="Unsupported density domain"):
        mb.view.density(front, domain="clinic", show=False)


def test_unknown_density_domain_not_silently_reinterpreted():
    front = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="Unsupported density domain"):
        mb.view.density(front, domain="typo", show=False)


def test_existing_statistics_contracts_remain_public():
    assert hasattr(mb.stats, "ranks")
    assert hasattr(mb.stats, "strata")
    assert hasattr(mb.stats, "tiers")
    assert not hasattr(mb.stats, "tier")
    assert not hasattr(mb.stats, "caste")
    assert not hasattr(mb, "Experiment")


def test_dtlz2_analytical_gt_density_and_invariants():
    mop = mb.mops.DTLZ2(M=10)
    gt = np.asarray(mop.pf(n_points=2000))
    assert gt.shape == (2000, 10)
    assert np.isfinite(gt).all()
    np.testing.assert_allclose(np.sum(gt**2, axis=1), 1.0, atol=1e-12, rtol=0.0)


def test_dpf4_high_density_gt_available_independently_of_fair():
    mop = mb.mops.DPF4(M=3)
    gt = np.asarray(mop.pf(n_points=10000))
    assert gt.shape == (10000, 3)
    assert np.isfinite(gt).all()
