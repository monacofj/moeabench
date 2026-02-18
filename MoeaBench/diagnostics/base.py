# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional, Any
from ..core.base import Reportable

class DiagnosticValue(Reportable):
    """
    Base class for diagnostic values that support narrative reporting.
    Wraps a numeric result and provides a human-readable explanation.
    """
    def __init__(self, value: float, name: str, description: str = "", raw_data: Optional[Any] = None):
        self.value = value
        self.name = name
        self.description = description
        self.raw_data = raw_data

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return f"<{self.name}: {self.value:.4f} (call .report() for details)>"

    def report(self, **kwargs) -> str:
        """Returns a basic narrative report."""
        if kwargs.get('markdown', True):
            return f"### {self.name}\n- **Value**: {self.value:.4f}\n- **Insight**: {self.description}"
        return f"{self.name}\n  Value: {self.value:.4f}\n  Insight: {self.description}"
