# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .topo import (topo_shape, topo_bands, topo_gap, topo_density)
from .perf import (perf_history, perf_spread, perf_density)
from .strat import (strat_ranks, strat_caste, strat_tiers)
from .clinic import (clinic_ecdf, clinic_distribution, clinic_history, clinic_radar)

from .style import apply_style

# Initialize the MoeaBench visual identity (Ocean Palette)
apply_style()

# Aliases for legacy support (Topography & Performance)
spaceplot = topo_shape
timeplot = perf_history

__all__ = [
    "topo_shape", "topo_bands", "topo_gap", "topo_density",
    "perf_history", "perf_spread", "perf_density",
    "strat_ranks", "strat_caste", "strat_tiers",
    "clinic_ecdf", "clinic_distribution", "clinic_history", "clinic_radar",
    "spaceplot", "timeplot"
]
