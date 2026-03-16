# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Ensure project root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import moeabench as mb
from moeabench.view.style import apply_style

# Mock plt.show to prevent it from clearing the figure
plt.show = lambda: None
apply_style()

def save_output(output, path, dpi=150):
    fig = None
    if isinstance(output, tuple) and output:
        first = output[0]
        if hasattr(first, "savefig"):
            fig = first
    elif hasattr(output, "_static_figure"):
        fig = output._static_figure
    elif hasattr(output, "figure") and hasattr(output.figure, "savefig"):
        fig = output.figure
    elif hasattr(output, "savefig"):
        fig = output

    if fig is None:
        fig = plt.gcf()

    fig.savefig(path, bbox_inches='tight', dpi=dpi)

def generate():
    img_dir = os.path.dirname(__file__)
    
    # Setup two experiments to show interesting comparisons
    print("Running experiments for clinical assets...")
    
    # 1. Healthy Algorithm (NSGA-III on DTLZ2)
    exp1 = mb.experiment(mop=mb.mops.DTLZ2(M=3), moea=mb.moeas.NSGA3(population=100, seed=5))
    exp1.run(repeat=5, generations=50)
    
    # 2. Pathological Algorithm (SPEA2 on DTLZ2 with low population to force gaps/poor coverage)
    exp2 = mb.experiment(mop=mb.mops.DTLZ2(M=3), moea=mb.moeas.SPEA2(population=20, seed=11))
    exp2.run(repeat=3, generations=10)
    
    # --- Instrument 1: Radar ---
    print("Generating clinic_radar.png...")
    out = mb.view.radar(exp1, show=False)
    save_output(out, os.path.join(img_dir, "clinic_radar.png"))
    plt.close('all')
    
    # --- Instrument 2: ECDF ---
    print("Generating clinic_ecdf.png...")
    out = mb.view.ecdf(exp1, metric="closeness", show=False)
    save_output(out, os.path.join(img_dir, "clinic_ecdf.png"))
    plt.close('all')
    
    # --- Instrument 3: Distribution ---
    print("Generating clinic_distribution.png...")
    out = mb.view.density(exp1, domain='clinic', metric="closeness", show=False)
    save_output(out, os.path.join(img_dir, "clinic_distribution.png"))
    plt.close('all')
    
    # --- Instrument 4: History ---
    print("Generating clinic_history.png...")
    out = mb.view.history(exp1, domain='clinic', metric="closeness", show=False)
    save_output(out, os.path.join(img_dir, "clinic_history.png"))
    plt.close('all')
    
    print("\nClinical assets generated successfully.")

if __name__ == "__main__":
    generate()
