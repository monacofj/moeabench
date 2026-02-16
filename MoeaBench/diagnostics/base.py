# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from ..core.base import Reportable

class DiagnosticValue(Reportable):
    """
    Base class for diagnostic values that support narrative reporting.
    Wraps a numeric result and provides a human-readable explanation.
    """
    def __init__(self, value: float, name: str, description: str = ""):
        self.value = value
        self.name = name
        self.description = description

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return f"<{self.name}: {self.value:.4f} (call .report() for details)>"

    def report(self, **kwargs) -> str:
        """Returns a basic narrative report."""
        return f"### {self.name}\n- **Value**: {self.value:.4f}\n- **Insight**: {self.description}"
