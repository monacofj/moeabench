# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from moeabench.view.style import MB_PALETTE, MB_PALETTE2, MOEABENCH_PALETTE


def test_mb_palette2_reorders_first_four_colors():
    assert MB_PALETTE[:4] == ["#1F3A5F", "#4C956C", "#9E4F2F", "#6F42A6"]
    assert MB_PALETTE2[:4] == ["#1F3A5F", "#9E4F2F", "#6F42A6", "#4C956C"]


def test_default_public_palette_points_to_mb_palette2():
    assert MOEABENCH_PALETTE == MB_PALETTE2
