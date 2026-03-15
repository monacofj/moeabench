# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import warnings

import moeabench as mb


def _runtime_warnings(caught):
    return [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


def test_topology_infers_gt_for_experiment_like_inputs(paired_experiments):
    exp1, exp2 = paired_experiments

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot = mb.view.topology(exp1, exp2, mode="static", show=False)

    assert plot is not None
    assert not any("GT could not be inferred" in msg for msg in _runtime_warnings(caught))


def test_topology_warns_when_gt_cannot_be_inferred_from_attainment_surface(paired_experiments):
    exp1, _ = paired_experiments
    att = mb.stats.attainment(exp1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot = mb.view.topology(att, mode="static", show=False)

    assert plot is not None
    assert any("GT could not be inferred" in msg for msg in _runtime_warnings(caught))


def test_topology_accepts_explicit_gt_for_attainment_surface(paired_experiments, canonical_gt):
    exp1, _ = paired_experiments
    att = mb.stats.attainment(exp1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot = mb.view.topology(att, gt=canonical_gt, mode="static", show=False)

    assert plot is not None
    assert not any("GT could not be inferred" in msg for msg in _runtime_warnings(caught))
