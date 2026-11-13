# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .topo import (
    topology as _topology,
    bands as _bands,
    gap as _gap,
    density as _topo_density,
)
from .perf import (
    perf_history as _perf_history,
    perf_spread as _spread,
    perf_density as _perf_density,
)
from .strat import (
    strat_ranks as _ranks,
    strat_strata as _strata,
    strat_tiers as _tiers,
)
import numpy as _np

from .style import apply_style

# Initialize the moeabench visual identity (Ocean Palette)
apply_style()

# Canonical API (chart-type oriented)
topology = _topology
bands = _bands
gap = _gap
spread = _spread
ranks = _ranks
strata = _strata
tiers = _tiers


def _resolve_view_domain(args, kwargs):
    """Resolve view domain for canonical dispatchers."""
    domain = kwargs.pop("domain", "auto")
    if domain in ("perf", "topo"):
        return domain
    if not args:
        return "perf"

    target = args[0]
    if (
        hasattr(target, "objectives")
        or hasattr(target, "results")
        or hasattr(target, "axes")
        or hasattr(target, "space")
    ):
        return "topo"
    if isinstance(target, _np.ndarray) and getattr(target, "ndim", 0) == 2:
        return "topo"
    if "space" in kwargs or "axes" in kwargs or "layout" in kwargs:
        return "topo"
    return "perf"


def density(*args, **kwargs):
    """Canonical density plot dispatcher (performance/topography)."""
    domain = _resolve_view_domain(args, kwargs)
    if domain == "clinic":
        raise ValueError("Clinical diagnostics are not available in this release.")
    if domain == "topo":
        return _topo_density(*args, **kwargs)
    return _perf_density(*args, **kwargs)


def history(*args, **kwargs):
    """Canonical history plot dispatcher (performance)."""
    domain = kwargs.pop("domain", "auto")
    if domain not in ("auto", "perf"):
        raise ValueError(f"Unsupported history domain: {domain}")
    if domain == "perf":
        return _perf_history(*args, **kwargs)

    return _perf_history(*args, **kwargs)

__all__ = [
    "topology", "bands", "gap", "density", "history", "spread",
    "ranks", "strata", "tiers"
]

for _name in ("topo", "perf", "strat", "style", "np"):
    globals().pop(_name, None)
