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

from .auditor import audit, fair_audit, q_audit, DiagnosticResult, FairAuditResult, QualityAuditResult
from .enums import DiagnosticStatus

# Fair Metrics (Physical Layer)
from .fair import (
    fair_headway,
    fair_closeness,
    fair_coverage,
    fair_gap,
    fair_regularity,
    fair_balance
)

# Q-Scores (Clinical Layer)
from .qscore import (
    q_headway,
    q_closeness,
    q_coverage,
    q_gap,
    q_regularity,
    q_balance,
    q_headway_points,
    q_closeness_points
)

__all__ = [
    "audit", "fair_audit", "q_audit", 
    "DiagnosticResult", "FairAuditResult", "QualityAuditResult",
    "DiagnosticStatus",
    "fair_headway", "fair_closeness", "fair_coverage", "fair_gap", "fair_regularity", "fair_balance",
    "q_headway", "q_closeness", "q_coverage", "q_gap", "q_regularity", "q_balance",
    "q_headway_points", "q_closeness_points"
]
