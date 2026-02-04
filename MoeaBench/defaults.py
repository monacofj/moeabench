# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Centralized defaults management for MoeaBench.
This module provides the `mb.defaults` object, allowing users to configure 
global settings for execution, statistics, and visualization.
"""

class Defaults:
    """
    Singleton-like object to manage global defaults for MoeaBench.
    Users can access and modify these values directly:
    >>> import MoeaBench as mb
    >>> mb.defaults.population = 200
    """
    def __init__(self):
        # --- Execution Defaults ---
        self.population = 150
        self.generations = 300
        self.seed = 1
        
        # --- Statistics Defaults ---
        self.alpha = 0.05                 # Default significance level for hypothesis tests
        self.cv_tolerance = 0.05          # Coefficient of Variation threshold for 'High' stability
        self.cv_moderate = 0.15           # Coefficient of Variation threshold for 'Low' stability
        self.displacement_threshold = 0.1  # Relative frequency threshold for displacement depth
        self.large_gap_threshold = 2      # Rank gap count considered 'Large' in reports
        
        # A12 Effect Size Thresholds (Vargha & Delaney, 2000)
        self.a12_negligible = 0.147
        self.a12_small = 0.33
        self.a12_medium = 0.474
        
        # --- Reporting & Visuals ---
        self.precision = 4                # Decimal places in narrative reports
        self.theme = 'moeabench'          # Default visualization color palette
        self.backend = 'auto'              # 'auto', 'plotly', or 'matplotlib'
        self.save_format = None            # Automatically save plots as 'pdf', 'png', etc. if set
        self.dpi = 300                     # Resolution for static image export
        self.figsize = (10, 8)            # Matplotlib figure size (width, height) in inches
        self.plot_width = 900             # Interactive (Plotly) chart width
        self.plot_height = 800            # Interactive (Plotly) chart height

    def __repr__(self):
        """Returns a clean narrative of current defaults."""
        lines = ["--- MoeaBench Defaults ---"]
        for k, v in vars(self).items():
            lines.append(f"  {k:<22}: {v}")
        return "\n".join(lines)

# Registry instance
defaults = Defaults()
