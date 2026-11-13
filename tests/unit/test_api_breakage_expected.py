# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import moeabench as mb


def test_experimental_diagnostics_are_not_public():
    assert not hasattr(mb, "clinic")
    assert not hasattr(mb, "diagnostics")
    assert not hasattr(mb.view, "radar")
    assert not hasattr(mb.view, "ecdf")


def test_removed_tier_public_stats_api():
    assert not hasattr(mb.stats, "tier")
