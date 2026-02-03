# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2026 Silva F. F. <fernandoferreira.silva42@usp.br>
#
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

from MoeaBench import mb
from MoeaBench.view.style import apply_style

# Mock plt.show to prevent it from clearing the figure
plt.show = lambda: None
apply_style()

def generate():
    # Save images in the current directory (docs/images/)
    img_dir = os.path.dirname(__file__)
    
    # --- 1. HELLO WORLD (Single Run Accuracy) ---
    print("Generating HELLO WORLD scenarios (repeat=1)...")
    hello = mb.experiment()
    hello.mop = mb.mops.DTLZ2(M=3)
    hello.moea = mb.moeas.NSGA3(population=100, generations=30)
    hello.run(repeat=1)
    
    print("Saving hello_space.png...")
    mb.view.spaceplot(hello, mode='static')
    plt.savefig(os.path.join(img_dir, "hello_space.png"), bbox_inches='tight', dpi=150)
    plt.close('all')
    
    print("Saving hello_time.png...")
    mb.view.timeplot(hello, mode='static')
    plt.savefig(os.path.join(img_dir, "hello_time.png"), bbox_inches='tight', dpi=150)
    plt.close('all')
    
    # --- 2. ADVANCED ANALYSIS (Scientific Scenarios - 5 Runs) ---
    print("\nGenerating ADVANCED scenarios (repeat=5)...")
    exp = mb.experiment()
    exp.name = "NSGA3"
    exp.mop = mb.mops.DTLZ2(M=3)
    exp.moea = mb.moeas.NSGA3(population=100, generations=30)
    exp.run(repeat=5)
    
    # Convergence cloud for more advanced sections if needed
    print("Saving timeplot.png (Cloud)...")
    mb.view.timeplot(exp, mode='static')
    plt.savefig(os.path.join(img_dir, "timeplot.png"), bbox_inches='tight', dpi=150)
    plt.close('all')
    
    # Diagnostics at G=10
    run = exp.last_run
    res_strata = mb.stats.strata(run.pop(10))
    
    print("Saving diagnostics (rank/caste)...")
    mb.view.rankplot(res_strata)
    plt.savefig(os.path.join(img_dir, "rankplot.png"), bbox_inches='tight', dpi=150)
    plt.close('all')
    
    mb.view.casteplot(res_strata)
    plt.savefig(os.path.join(img_dir, "casteplot.png"), bbox_inches='tight', dpi=150)
    plt.close('all')
    
    # Tier Duel
    print("Saving tierplot.png...")
    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.SPEA2(population=100, generations=30)
    exp2.run(repeat=5)
    mb.view.tierplot(exp, exp2, gen=10)
    plt.savefig(os.path.join(img_dir, "tierplot.png"), bbox_inches='tight', dpi=150)
    plt.close('all')

    # --- 3. MISSING ILLUSTRATIONS (Added v0.7.6+) ---
    
    # topo_bands
    print("Saving topo_bands.png...")
    mb.view.topo_bands(exp, levels=[0.5, 0.9])
    plt.savefig(os.path.join(img_dir, "topo_bands.png"), bbox_inches='tight', dpi=150)
    plt.close('all')

    # topo_gap
    print("Saving topo_gap.png...")
    mb.view.topo_gap(exp, exp2, level=0.5)
    plt.savefig(os.path.join(img_dir, "topo_gap.png"), bbox_inches='tight', dpi=150)
    plt.close('all')

    # topo_density (Axes 0 and 1)
    print("Saving topo_density.png...")
    mb.view.topo_density(exp, exp2, axes=[0, 1])
    plt.savefig(os.path.join(img_dir, "topo_density.png"), bbox_inches='tight', dpi=150)
    plt.close('all')

    # perf_spread (Hypervolume)
    print("Saving perf_spread.png...")
    mb.view.perf_spread(exp, exp2, metric=mb.metrics.hv)
    plt.savefig(os.path.join(img_dir, "perf_spread.png"), bbox_inches='tight', dpi=150)
    plt.close('all')

    # perf_density (Hypervolume)
    print("Saving perf_density.png...")
    mb.view.perf_density(exp, exp2, metric=mb.metrics.hv)
    plt.savefig(os.path.join(img_dir, "perf_density.png"), bbox_inches='tight', dpi=150)
    plt.close('all')
    
    print("\nAll assets generated successfully (Single vs. Multi-run distinction applied).")

if __name__ == "__main__":
    generate()
