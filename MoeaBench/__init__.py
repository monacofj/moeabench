# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Core Engine
from .core.experiment import experiment
from .core.run import Run
from .core.base_moea import BaseMoea

# Submodules
from . import benchmarks
from . import moeas
from . import metrics

# Plotting High-Level
from .plotting.plotter import spaceplot, timeplot

# MB Wrapper for Legacy Compatibility
class _MB:
    """
    Wrapper class to provide easy access via 'mb' object.
    """
    def __init__(self):
        self.experiment = experiment
        self.Run = Run
        self.benchmarks = benchmarks
        self.moeas = moeas
        self.metrics = metrics
        self.spaceplot = spaceplot
        self.timeplot = timeplot
        
        # Metric Shortcuts
        self.hv = metrics.hypervolume
        self.igd = metrics.igd
        self.gd = metrics.gd
        self.gdplus = metrics.gdplus
        self.igdplus = metrics.igdplus

mb = _MB()

__all__ = ["experiment", "Run", "benchmarks", "moeas", "metrics", "spaceplot", "timeplot", "mb"]
