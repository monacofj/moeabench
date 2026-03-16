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

import moeabench as mb
from moeabench.view.style import apply_style

# Mock plt.show to prevent it from clearing the figure
plt.show = lambda: None
apply_style()

def save_output(output, path, dpi=150):
    """Save a matplotlib/plot wrapper output produced by the current API."""
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
    # Save images in the current directory (docs/images/)
    img_dir = os.path.dirname(__file__)
    
    # --- 1. HELLO WORLD (Single Run Accuracy) ---
    print("Generating HELLO WORLD scenarios (repeat=1)...")
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ2()
    exp.moea = mb.moeas.NSGA3(seed=7)
    exp.run()
    
    print("Saving hello_space.png...")
    topo = mb.view.topology(exp, mode='static', show=False)
    save_output(topo, os.path.join(img_dir, "hello_space.png"))
    plt.close('all')

    print("Saving hello_decision.png...")
    topo_vars = mb.view.topology(
        exp.set(),
        mode='static',
        title="Decision-Space Geometry",
        axis_labels="Decision Variable",
        show_gt=False,
        show=False,
    )
    save_output(topo_vars, os.path.join(img_dir, "hello_decision.png"))
    plt.close('all')

    print("Saving hello_time.png...")
    out = mb.view.history(exp, mode='static', show=False)
    save_output(out, os.path.join(img_dir, "hello_time.png"))
    plt.close('all')
    
    # --- 2. ADVANCED ANALYSIS (Scientific Scenarios - 5 Runs) ---
    print("\nGenerating ADVANCED scenarios (repeat=5)...")
    exp = mb.experiment()
    exp.name = "NSGA3"
    exp.mop = mb.mops.DTLZ2(M=3)
    exp.moea = mb.moeas.NSGA3(population=100, generations=30, seed=2)
    exp.run(repeat=5)
    
    # Multi-run convergence history
    print("Saving timeplot.png...")
    out = mb.view.history(exp, mode='static', show=False)
    save_output(out, os.path.join(img_dir, "timeplot.png"))
    plt.close('all')

    # Diagnostics: Use MOEA/D on DTLZ1 to show interesting rank structures
    # (NSGA-III converges too fast on DTLZ2, collapsing ranks)
    print("Generating rankplot scenario (MOEA/D on DTLZ1)...")
    exp_rank = mb.experiment()
    exp_rank.mop = mb.mops.DTLZ1(M=3)
    exp_rank.moea = mb.moeas.MOEAD(population=100, generations=20, seed=1)
    exp_rank.run(repeat=1)
    res_layer = mb.stats.ranks(exp_rank.last_run.pop(14))
    strata_view = mb.stats.strata(exp_rank.last_run.pop(14))
    
    print("Saving diagnostics (rank/strata)...")
    out = mb.view.ranks(res_layer, show=False)
    save_output(out, os.path.join(img_dir, "rankplot.png"))
    plt.close('all')
    
    strata_ind = mb.stats.strata(exp, mode='individual')
    out = mb.view.strata(strata_ind, title="Population Merit", show=False)
    save_output(out, os.path.join(img_dir, "caste_individual.png"))
    plt.close('all')

    strata_coll = mb.stats.strata(exp, mode='collective')
    out = mb.view.strata(strata_coll, title="Stochastic Robustness", show=False)
    save_output(out, os.path.join(img_dir, "caste_collective.png"))
    plt.close('all')

    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.SPEA2(population=100, generations=30, seed=3)
    exp2.run(repeat=5)

    # --- 3. MISSING ILLUSTRATIONS (Added v0.7.6+) ---
    
    # bands
    print("Saving topo_bands.png...")
    out = mb.view.bands(exp, levels=[0.5, 0.9], show=False)
    save_output(out, os.path.join(img_dir, "topo_bands.png"))
    plt.close('all')

    # gap
    print("Saving topo_gap.png...")
    out = mb.view.gap(exp, exp2, level=0.5, show=False)
    save_output(out, os.path.join(img_dir, "topo_gap.png"))
    plt.close('all')

    # density (topology domain, axes 0 and 1)
    print("Saving topo_density.png...")
    out = mb.view.density(exp, exp2, domain='topo', axes=[0, 1], show=False)
    save_output(out, os.path.join(img_dir, "topo_density.png"))
    plt.close('all')

    # spread (Hypervolume)
    print("Saving perf_spread.png...")
    hv1 = mb.metrics.hv(exp)
    hv2 = mb.metrics.hv(exp2)
    out = mb.view.spread(hv1, hv2, show=False)
    save_output(out, os.path.join(img_dir, "perf_spread.png"))
    plt.close('all')

    # density (performance domain, Hypervolume)
    print("Saving perf_density.png...")
    out = mb.view.density(hv1, hv2, domain='perf', show=False)
    save_output(out, os.path.join(img_dir, "perf_density.png"))
    plt.close('all')
    
    print("\nAll assets generated successfully (Single vs. Multi-run distinction applied).")

if __name__ == "__main__":
    generate()
