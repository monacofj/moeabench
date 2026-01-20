# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Core Engine
from .core.experiment import experiment
from .core.run import Run
from .core.base_moea import BaseMoea

# Submodules
from . import mops
from . import moeas
from . import metrics
from . import stats
from . import view
from . import system

# MB Wrapper for direct access
class _MB:
    """
    Wrapper class to provide easy access via 'mb' object.
    """
    def __init__(self):
        self.experiment = experiment
        self.Experiment = experiment
        self.Run = Run
        self.mops = mops
        self.moeas = moeas
        self.metrics = metrics
        self.stats = stats
        self.view = view
        self.system = system
        
        # Taxonomy: Performance | Topography | Stratification (v0.7.0)
        self.perf_evidence = stats.perf_evidence
        self.perf_probability = stats.perf_probability
        self.perf_distribution = stats.perf_distribution
        self.topo_distribution = stats.topo_distribution
        self.topo_attainment = stats.topo_attainment
        self.topo_gap = stats.topo_gap
        
        # Aliases for legacy support (v0.6.x)
        self.perf_prob = stats.perf_probability
        self.perf_dist = stats.perf_distribution
        self.topo_dist = stats.topo_distribution
        self.topo_attain = stats.topo_attainment

        # Semantic View Shortcuts
        self.topo_shape = view.topo_shape
        self.topo_bands = view.topo_bands
        self.perf_history = view.perf_history
        self.perf_spread = view.perf_spread
        self.strat_caste = view.strat_caste
        
        # Legacy Delegates
        self.spaceplot = view.spaceplot
        self.timeplot = view.timeplot
        self.rankplot = view.rankplot
        self.casteplot = view.strat_caste
        self.tierplot = view.tierplot

mb = _MB()

__all__ = ["experiment", "Run", "mops", "moeas", "metrics", "stats", "view", "system", "mb"]
