# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import moeabench as mb


def test_public_namespaces():
    assert hasattr(mb, "clinic"), "Expected mb.clinic namespace"
    assert hasattr(mb, "stats"), "Expected mb.stats namespace"
    assert hasattr(mb, "view"), "Expected mb.view namespace"
    assert not hasattr(mb, "diagnostics"), "diagnostics should not leak at top level"
    assert not hasattr(mb, "core"), "core should not leak at top level"
    assert not hasattr(mb, "Experiment"), "mb.Experiment should not be part of canonical public API"


def test_clinic_surface():
    assert hasattr(mb.clinic, "audit"), "Expected clinic.audit"
    assert not hasattr(mb.clinic, "fair_audit"), "fair_audit should be removed"
    assert not hasattr(mb.clinic, "q_audit"), "q_audit should be removed"


def test_stats_compare_surface():
    assert hasattr(mb.stats, "perf_compare"), "Expected stats.perf_compare"
    assert hasattr(mb.stats, "topo_compare"), "Expected stats.topo_compare"
    assert hasattr(mb.stats, "ranks"), "Expected stats.ranks"
    assert hasattr(mb.stats, "strata"), "Expected stats.strata"
    assert hasattr(mb.stats, "tiers"), "Expected stats.tiers"
    assert not hasattr(mb.stats, "tier"), "tier should be removed in unified strata API"


def test_view_surface():
    for name in ("topology", "bands", "gap", "density", "history", "spread", "ranks", "strata", "tiers", "radar", "ecdf"):
        assert hasattr(mb.view, name), f"Expected mb.view.{name}"
    assert not hasattr(mb.view, "caste"), "view.caste should not be part of canonical public API"
    for hidden in ("topo", "perf", "strat", "clinic", "style", "np"):
        assert not hasattr(mb.view, hidden), f"view.{hidden} should not leak"


def test_system_surface():
    for name in ("version", "check_dependencies", "info", "output", "export_objectives", "export_variables"):
        assert hasattr(mb.system, name), f"Expected mb.system.{name}"
    for hidden in ("reproducibility_info", "emit_output", "importlib"):
        assert not hasattr(mb.system, hidden), f"system.{hidden} should not leak"
