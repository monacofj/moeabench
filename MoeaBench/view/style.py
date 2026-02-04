# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib.pyplot as plt
from cycler import cycler
import plotly.io as pio
import plotly.graph_objects as go
from ..defaults import defaults

# --- MoeaBench Standard Palette ---
# 1. Indigo:          #1A237E
# 2. Bordeaux:        #A00028 (Lighter)
# 3. Vibrant Jade:    #00BFA5
# 4. Soft Plum:       #8E24AA
# 5. Vivid Green:     #00E676
# 6. Intense Cyan:    #00E5FF
# 7. Orange:          #FF9100
# 8. Vivid Blue:      #2979FF
# 9. Yellow:          #FBC02D

MOEABENCH_PALETTE = [
    "#1A237E", "#A00028", "#00BFA5", 
    "#8E24AA", "#00E676", "#00E5FF",
    "#FF9100", "#2979FF", "#FBC02D"
]

def apply_style(theme=None):
    """
    Applies the MoeaBench standard style globally to Matplotlib and Plotly.
    """
    theme = theme if theme is not None else defaults.theme
    
    # Map themes
    if theme == 'moeabench':
        palette = MOEABENCH_PALETTE
    else:
        # Fallback to matplotlib default or other predefined if added
        palette = MOEABENCH_PALETTE 

    # 1. Matplotlib Global Configuration
    plt.rcParams['axes.prop_cycle'] = cycler(color=palette)
    plt.rcParams['grid.color'] = '#E0E0E0'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    
    # 2. Plotly Global Template Configuration
    # We create a custom 'moeabench' template based on 'plotly_white'
    mb_template = go.layout.Template(pio.templates["plotly_white"])
    mb_template.layout.colorway = palette
    
    # Add to Plotly registry and set as default
    pio.templates["moeabench"] = mb_template
    pio.templates.default = "moeabench"
