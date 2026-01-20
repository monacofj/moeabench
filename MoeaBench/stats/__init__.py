# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .tests import (perf_probability, perf_evidence, perf_distribution, topo_distribution)
from .stratification import strata, emd, StratificationResult, tier, TierResult
from .topo_attainment import (AttainmentSurface, topo_attainment, topo_gap)

# Aliases for legacy support (v0.6.0 -> v0.7.0)
perf_prob = perf_probability
perf_dist = perf_distribution
topo_dist = topo_distribution
topo_attain = topo_attainment
