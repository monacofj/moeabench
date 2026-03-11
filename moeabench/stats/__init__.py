# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .tests import perf_compare, topo_compare, PerfCompareResult
from .stratification import strata, emd, StratificationResult
from .topo_attainment import AttainmentSurface, topo_attainment as _attainment, topo_gap as _attainment_gap

# Canonical API names
attainment = _attainment
attainment_gap = _attainment_gap

# Method aliases for compare APIs.
def perf_shift(data1, data2, **kwargs):
    """Alias for perf_compare(method='shift')."""
    return perf_compare(data1, data2, method="shift", **kwargs)


def perf_match(data1, data2, **kwargs):
    """Alias for perf_compare(method='match')."""
    return perf_compare(data1, data2, method="match", **kwargs)


def perf_win(data1, data2, **kwargs):
    """Alias for perf_compare(method='win')."""
    return perf_compare(data1, data2, method="win", **kwargs)


def topo_match(*args, **kwargs):
    """Alias for topo_compare(method='match')."""
    return topo_compare(*args, method="match", **kwargs)


def topo_emd(*args, **kwargs):
    """Alias for topo_compare(method='emd')."""
    return topo_compare(*args, method="emd", **kwargs)


def topo_anderson(*args, **kwargs):
    """Alias for topo_compare(method='anderson')."""
    return topo_compare(*args, method="anderson", **kwargs)


# Prevent implicit submodule leakage (legacy public name).
globals().pop("topo_attainment", None)

__all__ = [
    "perf_compare",
    "topo_compare",
    "perf_shift",
    "perf_match",
    "perf_win",
    "topo_match",
    "topo_emd",
    "topo_anderson",
    "PerfCompareResult",
    "strata",
    "emd",
    "StratificationResult",
    "AttainmentSurface",
    "attainment",
    "attainment_gap",
]
