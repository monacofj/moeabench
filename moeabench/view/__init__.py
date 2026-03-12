# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .topo import (
    topo_shape as _topology,
    topo_bands as _bands,
    topo_gap as _gap,
    topo_density as _topo_density,
)
from .perf import (
    perf_history as _perf_history,
    perf_spread as _spread,
    perf_density as _perf_density,
)
from .strat import (
    strat_ranks as _ranks,
    strat_caste as _caste,
    strat_tiers as _tiers,
)
from .clinic import (
    clinic_ecdf as _ecdf,
    clinic_distribution as _clinic_density,
    clinic_history as _clinic_history,
    clinic_radar as _radar,
)
import numpy as np

from .style import apply_style

# Initialize the moeabench visual identity (Ocean Palette)
apply_style()

# Canonical API (chart-type oriented)
topology = _topology
bands = _bands
gap = _gap
spread = _spread
ranks = _ranks
caste = _caste
tiers = _tiers
ecdf = _ecdf
radar = _radar


def _resolve_view_domain(args, kwargs):
    """Resolve view domain for canonical dispatchers."""
    domain = kwargs.pop("domain", "auto")
    if domain in ("clinic", "perf", "topo"):
        return domain
    if not args:
        return "perf"

    target = args[0]
    clinic_metrics = {"closeness", "headway", "coverage", "gap", "regularity", "balance"}
    metric = kwargs.get("metric", None)
    if isinstance(metric, str) and metric.lower() in clinic_metrics:
        return "clinic"
    if hasattr(target, "q_audit_res") or hasattr(target, "fair_audit_res") or hasattr(target, "scores"):
        return "clinic"
    if hasattr(target, "objectives"):
        return "topo"
    if isinstance(target, np.ndarray) and getattr(target, "ndim", 0) == 2:
        return "topo"
    if "space" in kwargs or "axes" in kwargs or "layout" in kwargs:
        return "topo"
    return "perf"


def density(*args, **kwargs):
    """Canonical density plot dispatcher (clinic/perf/topo)."""
    domain = _resolve_view_domain(args, kwargs)
    if domain == "clinic":
        return _clinic_density(*args, **kwargs)
    if domain == "topo":
        return _topo_density(*args, **kwargs)
    return _perf_density(*args, **kwargs)


def history(*args, **kwargs):
    """Canonical history plot dispatcher (clinic/perf)."""
    domain = kwargs.pop("domain", "auto")
    if domain == "clinic":
        return _clinic_history(*args, **kwargs)
    if domain == "perf":
        return _perf_history(*args, **kwargs)

    # Auto: clinic if metric is a clinical metric or target is clinical result.
    clinic_metrics = {"closeness", "headway", "coverage", "gap", "regularity", "balance"}
    metric = kwargs.get("metric", None)
    if isinstance(metric, str) and metric.lower() in clinic_metrics:
        return _clinic_history(*args, **kwargs)
    if args and (hasattr(args[0], "q_audit_res") or hasattr(args[0], "fair_audit_res")):
        return _clinic_history(*args, **kwargs)
    return _perf_history(*args, **kwargs)

__all__ = [
    "topology", "bands", "gap", "density", "history", "spread",
    "ranks", "caste", "tiers", "ecdf", "radar"
]
