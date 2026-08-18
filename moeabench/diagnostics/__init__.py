# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
moeabench Diagnostics Module
============================

Provides clinical diagnostics for evolutionary algorithm performance.
Divided into:
- Physical Metrics (Physical, Scale-corrected): mb.diagnostics.headway, etc.
- Q-Scores (Clinical, Calibration-corrected): mb.diagnostics.q_headway, etc.
- Calibration (Plugin Support): mb.diagnostics.calibrate, mb.diagnostics.register_baselines
"""

from .auditor import audit, PerformanceAuditor, DiagnosticResult, FairAuditResult, QualityAuditResult
from .enums import DiagnosticStatus
from .baselines import register_baselines, reset_baselines, use_baselines
from .calibration import calibrate_mop as calibrate

# FAIR Metrics (Physical Layer)
from .fair import (
    headway as _fair_headway,
    closeness as _fair_closeness,
    coverage as _fair_coverage,
    gap as _fair_gap,
    regularity as _fair_regularity,
    balance as _fair_balance,
)

# Q-Scores (Clinical Layer)
from .qscore import (
    q_headway as _q_headway,
    q_closeness as _q_closeness,
    q_coverage as _q_coverage,
    q_gap as _q_gap,
    q_regularity as _q_regularity,
    q_balance as _q_balance,
    q_headway_points,
    q_closeness_points
)


def _is_experiment_or_run(data):
    return hasattr(data, "pop") and callable(data.pop)


def _canonical_fair_metric(name, data, direct_fn):
    """Return the exact physical metric used by ``audit(data)``."""
    diagnostic = audit(data, quality=False)
    result = diagnostic.fair.metrics[name]
    ctx = diagnostic.diagnostic_context or {}

    if name in ("HEADWAY", "CLOSENESS"):
        history_result = direct_fn(data, ctx.get("GT"), ctx.get("s_k"),
                                   problem=ctx.get("problem"), k=ctx.get("k"))
    elif name in ("COVERAGE", "GAP"):
        history_result = direct_fn(data, ctx.get("GT"),
                                   problem=ctx.get("problem"), k=ctx.get("k"))
    elif name == "REGULARITY":
        history_result = direct_fn(data, ctx.get("U_ref"),
                                   problem=ctx.get("problem"), k=ctx.get("k"))
    else:
        history_result = direct_fn(data, ctx.get("centroids"), ctx.get("ref_hist"),
                                   problem=ctx.get("problem"), k=ctx.get("k"))

    if history_result.history_values is not None:
        result._history_values = history_result.history_values
        result._history_labels = history_result.history_labels
    return result


def headway(data, ref=None, s_k=None, **kwargs):
    if _is_experiment_or_run(data) and ref is None and s_k is None and not kwargs:
        return _canonical_fair_metric("HEADWAY", data, _fair_headway)
    return _fair_headway(data, ref, s_k, **kwargs)


def closeness(data, ref=None, s_k=None, **kwargs):
    if _is_experiment_or_run(data) and ref is None and s_k is None and not kwargs:
        return _canonical_fair_metric("CLOSENESS", data, _fair_closeness)
    return _fair_closeness(data, ref, s_k, **kwargs)


def coverage(data, ref=None, **kwargs):
    if _is_experiment_or_run(data) and ref is None and not kwargs:
        return _canonical_fair_metric("COVERAGE", data, _fair_coverage)
    return _fair_coverage(data, ref, **kwargs)


def gap(data, ref=None, **kwargs):
    if _is_experiment_or_run(data) and ref is None and not kwargs:
        return _canonical_fair_metric("GAP", data, _fair_gap)
    return _fair_gap(data, ref, **kwargs)


def regularity(data, ref_distribution=None, **kwargs):
    if _is_experiment_or_run(data) and ref_distribution is None and not kwargs:
        return _canonical_fair_metric("REGULARITY", data, _fair_regularity)
    return _fair_regularity(data, ref_distribution, **kwargs)


def balance(data, centroids=None, ref_hist=None, **kwargs):
    if _is_experiment_or_run(data) and centroids is None and ref_hist is None and not kwargs:
        return _canonical_fair_metric("BALANCE", data, _fair_balance)
    return _fair_balance(data, centroids, ref_hist, **kwargs)


def _canonical_q_score(name, data):
    return audit(data).quality.scores[name]


def q_headway(data, ref=None, s_k=None, **kwargs):
    if _is_experiment_or_run(data) and ref is None and s_k is None and not kwargs:
        return _canonical_q_score("Q_HEADWAY", data)
    return _q_headway(data, ref, s_k, **kwargs)


def q_closeness(data, ref=None, s_k=None, **kwargs):
    if _is_experiment_or_run(data) and ref is None and s_k is None and not kwargs:
        return _canonical_q_score("Q_CLOSENESS", data)
    return _q_closeness(data, ref, s_k, **kwargs)


def q_coverage(data, ref=None, **kwargs):
    if _is_experiment_or_run(data) and ref is None and not kwargs:
        return _canonical_q_score("Q_COVERAGE", data)
    return _q_coverage(data, ref, **kwargs)


def q_gap(data, ref=None, **kwargs):
    if _is_experiment_or_run(data) and ref is None and not kwargs:
        return _canonical_q_score("Q_GAP", data)
    return _q_gap(data, ref, **kwargs)


def q_regularity(data, ref_distribution=None, **kwargs):
    if _is_experiment_or_run(data) and ref_distribution is None and not kwargs:
        return _canonical_q_score("Q_REGULARITY", data)
    return _q_regularity(data, ref_distribution, **kwargs)


def q_balance(data, centroids=None, ref_hist=None, **kwargs):
    if _is_experiment_or_run(data) and centroids is None and ref_hist is None and not kwargs:
        return _canonical_q_score("Q_BALANCE", data)
    return _q_balance(data, centroids, ref_hist, **kwargs)

__all__ = [
    "audit", "PerformanceAuditor",
    "DiagnosticResult", "FairAuditResult", "QualityAuditResult",
    "DiagnosticStatus",
    "register_baselines", "reset_baselines", "use_baselines", "calibrate",
    "headway", "closeness", "coverage", "gap", "regularity", "balance",
    "q_headway", "q_closeness", "q_coverage", "q_gap", "q_regularity", "q_balance",
    "q_headway_points", "q_closeness_points"
]

for _name in ("auditor", "base", "baselines", "calibration", "enums", "fair", "qscore", "utils"):
    globals().pop(_name, None)
