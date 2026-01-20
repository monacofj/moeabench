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
        
        # Taxonomy: Performance | Topography | Stratification (v0.6.3)
        self.perf_evidence = stats.perf_evidence
        self.perf_prob = stats.perf_prob
        self.perf_dist = stats.perf_dist
        self.topo_dist = stats.topo_dist # Note: this is the stats version. 
        self.topo_attain = stats.topo_attain
        self.topo_gap = stats.topo_gap

        # Space/Time Plot Shortcuts (keeping basic non-taxonomy shortcuts)
        self.spaceplot = view.spaceplot
        self.timeplot = view.timeplot

mb = _MB()

__all__ = ["experiment", "Run", "mops", "moeas", "metrics", "stats", "view", "system", "mb"]
