# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
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
from . import system

# Plotting High-Level
from .plotting.plotter import spaceplot, timeplot, polarplot, profileplot

# MB Wrapper for Legacy Compatibility
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
        self.spaceplot = spaceplot
        self.timeplot = timeplot
        self.polarplot = polarplot
        self.profileplot = profileplot
        self.system = system
        
        # Metric Shortcuts
        self.hv = metrics.hypervolume
        self.igd = metrics.igd
        self.gd = metrics.gd
        self.gdplus = metrics.gdplus
        self.igdplus = metrics.igdplus
        
        # Stats Shortcuts
        self.stratification = stats.stratification
        self.stratification_plot = stats.stratification_plot
        self.emd = stats.emd
        self.attainment = stats.attainment
        self.attainment_diff = stats.attainment_diff
        
        # Benchmark Shortcuts (for didactics)
        self.DTLZ1 = mops.DTLZ1
        self.DTLZ2 = mops.DTLZ2
        self.DTLZ3 = mops.DTLZ3
        self.DTLZ4 = mops.DTLZ4
        self.DTLZ5 = mops.DTLZ5
        self.DTLZ7 = mops.DTLZ7
        self.DTLZ8 = mops.DTLZ8
        self.DTLZ9 = mops.DTLZ9
        self.DPF1 = mops.DPF1
        self.DPF2 = mops.DPF2
        self.DPF3 = mops.DPF3
        self.DPF4 = mops.DPF4
        self.DPF5 = mops.DPF5
        
        # Algorithm Shortcuts (for didactics)
        self.NSGA2 = moeas.NSGA2deap
        self.SPEA2 = moeas.SPEA2
        self.MOEAD = moeas.MOEAD
        self.RVEA = moeas.RVEA
        self.NSGA3 = moeas.NSGA3

mb = _MB()

__all__ = ["experiment", "Run", "mops", "moeas", "metrics", "stats", "system", "spaceplot", "timeplot", "mb"]
