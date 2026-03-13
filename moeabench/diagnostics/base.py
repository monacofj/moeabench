# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional, Any
import numpy as np
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

    @property
    def samples(self):
        """Raw numeric samples used by distributional views when available."""
        if self.raw_data is None:
            return np.array([float(self.value)])
        arr = np.asarray(self.raw_data, dtype=float).ravel()
        return arr[np.isfinite(arr)]

    @property
    def sorted_samples(self):
        data = self.samples
        return np.sort(data)

    @property
    def ecdf_y(self):
        data = self.sorted_samples
        if data.size == 0:
            return np.array([])
        return np.linspace(0, 1, len(data))

    @property
    def median(self):
        data = self.samples
        return float(np.median(data)) if data.size else float("nan")

    @property
    def p95(self):
        data = self.samples
        return float(np.percentile(data, 95)) if data.size else float("nan")

    @property
    def history_values(self):
        return getattr(self, "_history_values", None)

    @property
    def history_labels(self):
        return getattr(self, "_history_labels", None)

    def report(self, show: bool = True, **kwargs) -> str:
        """Returns a basic narrative report."""
        if kwargs.get('markdown', self._is_notebook()):
            content = f"### {self.name}\n- **Value**: {self.value:.4f}\n- **Insight**: {self.description}"
        else:
            content = f"{self.name}\n  Value: {self.value:.4f}\n  Insight: {self.description}"
        return self._render_report(content, show, **kwargs)
