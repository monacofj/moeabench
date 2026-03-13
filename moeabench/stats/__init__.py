# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .tests import perf_compare, topo_compare, PerfCompareResult
from .stratification import (
    ranks,
    strata,
    tiers,
    emd,
    RankCompareResult,
    StrataCompareResult,
    TierResult,
)
from .attainment import AttainmentSurface, attainment, attainment_gap

# Method aliases for compare APIs.
def perf_shift(data1, data2, **kwargs):
    """Alias for perf_compare(method='mannwhitney')."""
    return perf_compare(data1, data2, method="mannwhitney", **kwargs)


def perf_match(data1, data2, **kwargs):
    """Alias for perf_compare(method='ks')."""
    return perf_compare(data1, data2, method="ks", **kwargs)


def perf_win(data1, data2, **kwargs):
    """Alias for perf_compare(method='a12')."""
    return perf_compare(data1, data2, method="a12", **kwargs)


def topo_match(*args, **kwargs):
    """Alias for topo_compare(method='ks')."""
    return topo_compare(*args, method="ks", **kwargs)


def topo_shift(*args, **kwargs):
    """Alias for topo_compare(method='emd')."""
    return topo_compare(*args, method="emd", **kwargs)


def topo_tail(*args, **kwargs):
    """Alias for topo_compare(method='anderson')."""
    return topo_compare(*args, method="anderson", **kwargs)


__all__ = [
    "perf_compare",
    "topo_compare",
    "perf_shift",
    "perf_match",
    "perf_win",
    "topo_match",
    "topo_shift",
    "topo_tail",
    "PerfCompareResult",
    "ranks",
    "strata",
    "tiers",
    "emd",
    "RankCompareResult",
    "StrataCompareResult",
    "TierResult",
    "AttainmentSurface",
    "attainment",
    "attainment_gap",
]

for _name in ("tests", "base", "stratification"):
    globals().pop(_name, None)
