# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from ..core.base import Reportable

class StatsResult(Reportable):
    """
    Base class for statistical result objects in MoeaBench.
    Provides narrative reporting and consistent representation.
    """
    def report(self, **kwargs) -> str:
        """Returns a human-readable narrative report of the results."""
        raise NotImplementedError("Subclasses must implement .report()")

class SimpleStatsValue(StatsResult):
    """Wrapper for single numeric results (A12, EMD, etc.)."""
    def __init__(self, value, name):
        self.value = value
        self.name = name
        
    def __float__(self): return float(self.value)
    
    def report(self) -> str:
        return f"--- {self.name} ---\n  Value: {self.value:.4f}"
