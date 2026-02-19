# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os
import site

# --- Environment Fix: Matplotlib 3.10 + Conflict with system mpl_toolkits ---
# Some Linux environments (e.g., Debian/Ubuntu) can have older system-level 
# mpl_toolkits that conflict with user-installed Matplotlib 3.10+.
# We inject the user-site path into mpl_toolkits.__path__ to ensure compatibility.
try:
    import mpl_toolkits
    user_site = site.getusersitepackages()
    local_mpl = os.path.join(user_site, 'mpl_toolkits')
    if os.path.exists(local_mpl) and hasattr(mpl_toolkits, '__path__'):
        if local_mpl not in mpl_toolkits.__path__:
            mpl_toolkits.__path__.insert(0, local_mpl)
    # Trigger mplot3d early to ensure it works
    from mpl_toolkits import mplot3d
except (ImportError, AttributeError, Exception):
    pass # Fallback handled by plotting layer
# ----------------------------------------------------------------------------

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
from .defaults import defaults
from . import diagnostics

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
        self.defaults = defaults
        self.diagnostics = diagnostics
        self.calibrate = diagnostics.calibrate
        """
        [mb.calibrate]
        Entry point for MOP Calibration. 
        Calculates or loads clinical baselines for a problem instance.
        """
        
        self.register_baselines = diagnostics.register_baselines
        """
        [mb.register_baselines]
        Registers an external JSON sidecar or baseline dictionary in the current session.
        """
        
        # Taxonomy: Performance | Topography | Stratification (v0.7.0)
        self.perf_evidence = stats.perf_evidence
        self.perf_probability = stats.perf_probability
        self.perf_distribution = stats.perf_distribution
        self.topo_distribution = stats.topo_distribution
        self.topo_attainment = stats.topo_attainment
        self.topo_gap = stats.topo_gap
        
        
        # Semantic View Shortcuts
        self.topo_shape = view.topo_shape
        self.topo_bands = view.topo_bands
        self.perf_history = view.perf_history
        self.perf_spread = view.perf_spread
        self.strat_caste = view.strat_caste
        
        # Legacy Delegates (Supported Aliases)
        self.spaceplot = view.spaceplot
        self.timeplot = view.timeplot

mb = _MB()

__all__ = ["experiment", "Run", "mops", "moeas", "metrics", "stats", "view", "system", "mb"]
