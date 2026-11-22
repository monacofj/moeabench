# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit entrypoint for bounded original DTLZ5/DTLZ6 scalarization (#65).

The canonical numerical implementation lives in moeabench.mops._dtlz56_front.
Independent checks remain in dtlz56_fixed_g_dominance_oracle.py, the separate
continuous g=0 oracle and focused tests. Re-export here for research scripts.
"""

from moeabench.mops._dtlz56_front import (  # noqa: F401
    ScalarizedCandidate,
    _bounded_scale,
    _relaxed_feasible,
    decision_vector,
    sample_original_dtlz56_ps,
    scalarized_candidate,
)
