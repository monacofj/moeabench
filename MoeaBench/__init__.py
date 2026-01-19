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

# View Shortcuts (Exposed at top level for convenience)
from .view.plotters import spaceplot, timeplot, rankplot, casteplot, tierplot, distplot
import warnings

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
        self.view = view
        self.system = system
        
        # Helper for Deprecation Warnings
        def _warn_shortcut(fn_name, fn, target_ns):
            def wrapper(*args, **kwargs):
                warnings.warn(
                    f"mb.{fn_name}() is deprecated and will be removed in future releases. "
                    f"Please use mb.{target_ns}.{fn_name}() instead.",
                    UserWarning, stacklevel=2
                )
                return fn(*args, **kwargs)
            return wrapper

        # 1. View Shortcuts (Legacy Deprecated)
        self.spaceplot = _warn_shortcut("spaceplot", view.spaceplot, "view")
        self.timeplot = _warn_shortcut("timeplot", view.timeplot, "view")
        self.rankplot = _warn_shortcut("rankplot", view.rankplot, "view")
        self.casteplot = _warn_shortcut("casteplot", view.casteplot, "view")
        self.tierplot = _warn_shortcut("tierplot", view.tierplot, "view")
        self.distplot = _warn_shortcut("distplot", view.distplot, "view")
        
        # 2. Metric Shortcuts (Legacy Deprecated)
        self.hv = _warn_shortcut("hypervolume", metrics.hypervolume, "metrics")
        self.igd = _warn_shortcut("igd", metrics.igd, "metrics")
        self.gd = _warn_shortcut("gd", metrics.gd, "metrics")
        self.gdplus = _warn_shortcut("gdplus", metrics.gdplus, "metrics")
        self.igdplus = _warn_shortcut("igdplus", metrics.igdplus, "metrics")
        
        # 3. Stats Shortcuts (Legacy Deprecated)
        self.strata = _warn_shortcut("strata", stats.strata, "stats")
        self.emd = _warn_shortcut("emd", stats.emd, "stats")
        self.attainment = _warn_shortcut("attainment", stats.attainment, "stats")
        self.attainment_diff = _warn_shortcut("attainment_diff", stats.attainment_diff, "stats")
        self.tier = _warn_shortcut("tier", stats.tier, "stats")
        
        # 4. Benchmark Shortcuts (Legacy Deprecated)
        for mop_name in ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7", "DTLZ8", "DTLZ9",
                         "DPF1", "DPF2", "DPF3", "DPF4", "DPF5"]:
             if hasattr(mops, mop_name):
                 setattr(self, mop_name, _warn_shortcut(mop_name, getattr(mops, mop_name), "mops"))
        
        # 5. Algorithm Shortcuts (Legacy Deprecated)
        self.NSGA2 = _warn_shortcut("NSGA2", moeas.NSGA2deap, "moeas")
        self.SPEA2 = _warn_shortcut("SPEA2", moeas.SPEA2, "moeas")
        self.MOEAD = _warn_shortcut("MOEAD", moeas.MOEAD, "moeas")
        self.RVEA = _warn_shortcut("RVEA", moeas.RVEA, "moeas")
        self.NSGA3 = _warn_shortcut("NSGA3", moeas.NSGA3, "moeas")

mb = _MB()

__all__ = ["experiment", "Run", "mops", "moeas", "metrics", "stats", "view", "system", "mb"]
