# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent continuous-g dominance exclusion for original DTLZ5/6 (#65).

Not a finite-grid approximation: a relaxed outer enclosure of every entire
closed g interval is tested recursively. This certifies absence of dominating
original feasible points with strictly smaller g up to a stated g tolerance.
The separate scalarization accuracy bound remains tested independently.
"""

import numpy as np
import pytest

from moeabench.mops import DTLZ5, DTLZ6


def _optimistic_interval_has_a_dominator(y, g_low, g_high):
    """Necessary box-feasibility test for ANY original g in [low,high].

    We optimistically give this interval its smallest possible radius,
    r=1+g_low, and widest angular feasible interval, attained at g_high.
    If even this superset cannot dominate y, no original point in the
    entire continuous interval can dominate y.
    """
    radius = 1.0 + g_low
    if y[-1] >= radius:
        return True
    remaining = np.sqrt(max(0.0, radius**2 - y[-1]**2))
    lo = np.pi / (4.0*(1.0+g_high))
    hi = np.pi*(1.0+2.0*g_high)/(4.0*(1.0+g_high))
    for j in range(len(y)-2, 0, -1):
        if remaining*np.sin(lo) > y[j] + 1e-12:
            return False
        used = min(y[j], remaining*np.sin(hi))
        remaining = np.sqrt(max(0.0, remaining**2 - used**2))
    return remaining <= y[0] + 1e-12


def _exclude_strict_smaller_g_dominators(y, g, *, g_tolerance=3e-5):
    """Certify the entire interval [0,g-g_tolerance], not discrete samples."""
    upper = g - g_tolerance
    if upper <= 0.0:
        return
    intervals = [(0.0, upper)]
    nodes = 0
    while intervals:
        low, high = intervals.pop()
        if not _optimistic_interval_has_a_dominator(y, low, high):
            continue
        nodes += 1
        assert nodes < 15000, "continuous-g global exclusion not resolved"
        assert high-low > 1e-12, "dominator remains possible on tiny interval"
        mid = (low+high)/2.0
        intervals.extend(((low, mid), (mid, high)))


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
@pytest.mark.parametrize("M", [4, 5, 10])
def test_no_dominator_anywhere_on_continuous_lower_g_interval(cls, M):
    """Independent continuous global nondominance for a diverse finite net."""
    mop = cls(M=M)
    ps = mop.ps(n_points=32)
    front = mop.evaluation(ps)["F"]
    tail = ps[:, M-1:]
    g = (
        np.sum((tail-0.5)**2, axis=1)
        if cls is DTLZ5 else np.sum(tail**0.1, axis=1)
    )
    # Test multiple nondegenerate rays, not solely the g=0 anchors.
    indices = np.flatnonzero(g > 1e-3)
    assert len(indices) >= 5
    selected = indices[np.linspace(0, len(indices)-1, 6, dtype=int)]
    for i in selected:
        _exclude_strict_smaller_g_dominators(front[i], g[i])
