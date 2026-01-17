# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib.pyplot as plt
from cycler import cycler
import plotly.io as pio
import plotly.graph_objects as go

# --- MoeaBench "Ocean" Palette ---
# 1. Indigo:          #1A237E
# 2. Emerald Green:   #2E7D32
# 3. Soft Plum:       #8E24AA
# 4. Vibrant Jade:    #00BFA5
# 5. Bordeaux:        #800020
# 6. Deep Teal/Cyan:  #006064
# 7. Orange:          #EF6C00
# 8. Red:             #C62828
# 9. Yellow:          #FBC02D

OCEAN_PALETTE = [
    "#1A237E", "#2E7D32", "#8E24AA", 
    "#00BFA5", "#800020", "#006064",
    "#EF6C00", "#C62828", "#FBC02D"
]

def apply_style():
    """
    Applies the MoeaBench "Ocean" style globally to Matplotlib and Plotly.
    """
    # 1. Matplotlib Global Configuration
    plt.rcParams['axes.prop_cycle'] = cycler(color=OCEAN_PALETTE)
    plt.rcParams['grid.color'] = '#E0E0E0'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    
    # 2. Plotly Global Template Configuration
    # We create a custom 'moeabench' template based on 'plotly_white'
    mb_template = go.layout.Template(pio.templates["plotly_white"])
    mb_template.layout.colorway = OCEAN_PALETTE
    
    # Add to Plotly registry and set as default
    pio.templates["moeabench"] = mb_template
    pio.templates.default = "moeabench"
