# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib.pyplot as plt
from cycler import cycler
import plotly.io as pio
import plotly.graph_objects as go
from ..defaults import defaults

# --- MoeaBench Official Palette (Original Order) ---
# 1. Navy Blue:       #1F3A5F
# 2. Muted Green:     #4C956C
# 3. Burnt Red:       #9E4F2F
# 4. Muted Violet:    #6F42A6
# 5. Amber:           #E09F3E
# 6. Bordeaux:        #7F1734
# 7. Sand:            #C7A66B
# 8. Blue Turquoise:  #2C7FB8

MB_PALETTE = [
    "#1F3A5F",
    "#4C956C",
    "#9E4F2F",
    "#6F42A6",
    "#E09F3E",
    "#7F1734",
    "#C7A66B",
    "#2C7FB8",
]

# --- MoeaBench Palette 2 (Default Order) ---
# 1. Navy Blue:       #1F3A5F
# 2. Burnt Red:       #9E4F2F
# 3. Muted Violet:    #6F42A6
# 4. Muted Green:     #4C956C
# 5. Amber:           #E09F3E
# 6. Bordeaux:        #7F1734
# 7. Sand:            #C7A66B
# 8. Blue Turquoise:  #2C7FB8

MB_PALETTE2 = [
    "#1F3A5F",
    "#9E4F2F",
    "#6F42A6",
    "#4C956C",
    "#E09F3E",
    "#7F1734",
    "#C7A66B",
    "#2C7FB8",
]

# Backwards-compatible public name: current default rotating palette.
MOEABENCH_PALETTE = MB_PALETTE2

# Semantic colors stay fixed across charts and do not rotate with the palette.
GT_COLOR = "#9CA3AF"
GRID_COLOR = "#D9DEE6"
TEXT_MUTED = "#6B7280"
MEDIAN_COLOR = "#8D99AE"
ALERT_COLOR = "#9E4F2F"

def apply_style(theme=None):
    """
    Applies the moeabench standard style globally to Matplotlib and Plotly.
    """
    theme = theme if theme is not None else defaults.theme
    
    # Map themes
    if theme in {'moeabench', 'mb_palette2', 'moeabench2'}:
        palette = MB_PALETTE2
    elif theme in {'mb_palette', 'moeabench-original', 'moeabench1'}:
        palette = MB_PALETTE
    else:
        # Fallback to matplotlib default or other predefined if added
        palette = MOEABENCH_PALETTE 

    # 1. Matplotlib Global Configuration
    plt.rcParams['axes.prop_cycle'] = cycler(color=palette)
    plt.rcParams['grid.color'] = GRID_COLOR
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.35
    
    # 2. Plotly Global Template Configuration
    # We create a custom 'moeabench' template based on 'plotly_white'
    mb_template = go.layout.Template(pio.templates["plotly_white"])
    mb_template.layout.colorway = palette
    mb_template.layout.xaxis.gridcolor = GRID_COLOR
    mb_template.layout.yaxis.gridcolor = GRID_COLOR
    mb_template.layout.font = dict(color="#1F2937")
    
    # Add to Plotly registry and set as default
    pio.templates["moeabench"] = mb_template
    pio.templates.default = "moeabench"
