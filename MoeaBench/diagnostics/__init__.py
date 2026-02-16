# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Diagnostics Module
============================

Provides clinical diagnostics for evolutionary algorithm performance.
Divided into:
- Fair Metrics (Physical, Scale-corrected): mb.diagnostics.fair_...
- Q-Scores (Clinical, Calibration-corrected): mb.diagnostics.q_...
"""

from .auditor import audit, DiagnosticResult
from .enums import DiagnosticStatus

# Fair Metrics (Physical Layer)
from .fair import (
    fair_denoise,
    fair_closeness,
    fair_coverage,
    fair_gap,
    fair_regularity,
    fair_balance
)

# Q-Scores (Clinical Layer)
from .qscore import (
    q_denoise,
    q_closeness,
    q_coverage,
    q_gap,
    q_regularity,
    q_balance,
    q_denoise_points,
    q_closeness_points
)

__all__ = [
    "audit", "DiagnosticResult", "DiagnosticStatus",
    "fair_denoise", "fair_closeness", "fair_coverage", "fair_gap", "fair_regularity", "fair_balance",
    "q_denoise", "q_closeness", "q_coverage", "q_gap", "q_regularity", "q_balance",
    "q_denoise_points", "q_closeness_points"
]
