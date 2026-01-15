# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

class StatsResult:
    """
    Base class for statistical result objects in MoeaBench.
    Provides narrative reporting and consistent representation.
    """
    def report(self) -> str:
        """Returns a human-readable narrative report of the results."""
        raise NotImplementedError("Subclasses must implement .report()")

    def __repr__(self):
        # We don't automatically call report() in __repr__ to avoid 
        # overwhelming the REPL, but we mention it.
        return f"<{self.__class__.__name__} (call .report() for details)>"

    def _repr_pretty_(self, p, cycle):
        """Rich representation for Jupyter/IPython."""
        if cycle:
            p.text(str(self))
            return
        p.text(self.report())

class SimpleStatsValue(StatsResult):
    """Wrapper for single numeric results (A12, EMD, etc.)."""
    def __init__(self, value, name):
        self.value = value
        self.name = name
        
    def __float__(self): return float(self.value)
    
    def report(self) -> str:
        return f"--- {self.name} ---\n  Value: {self.value:.4f}"
