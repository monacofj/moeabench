#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 10: Topological Equivalence (topo_dist)
-----------------------------------------------
This example demonstrates how to use the 'topo_dist' tool to determine 
if two different algorithms have converged to statistically equivalent 
distributions in both the objective and decision spaces.
"""

import mb_path
from MoeaBench import mb
import matplotlib.pyplot as plt

def main():
    print(f"Version: {mb.system.version()}")
    # 1. Setup: Compare NSGA-II vs NSGA-III on DTLZ2
    # We use a 3-objective problem (M=3)
    exp1 = mb.experiment()
    exp1.name = "NSGA-II"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA2(population=100, generations=50)

    exp2 = mb.experiment()
    exp2.name = "NSGA-III"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.NSGA3(population=100, generations=50)

    # 2. Execution: Run both experiments
    print("Running NSGA-II...")
    exp1.run()
    
    print("Running NSGA-III...")
    exp2.run()

    # 3. Topological Analysis: Objective Space
    # By default, topo_distribution(exp1, exp2) uses space='objs' and alpha=0.05
    print("\n--- [Analysis 1] Convergence Equivalence (Objective Space) ---")
    res_objs = mb.stats.topo_distribution(exp1, exp2, alpha=0.01) # Stricter alpha
    print(res_objs.report())

    # 4. Topological Analysis: Decision Space
    # We can also check if they found the same solutions in the decision space
    print("\n--- [Analysis 2] Strategy Equivalence (Decision Space) ---")
    res_vars = mb.stats.topo_distribution(exp1, exp2, space='vars')
    print(res_vars.report())

    # 5. Advanced: Earth Mover Distance (Geometric Distance)
    # Quantify "how far" the distributions are from each other
    print("\n--- [Analysis 3] Geometric Distance (EMD) ---")
    # Custom threshold: declare match only if distance < 0.05
    res_emd = mb.stats.topo_distribution(exp1, exp2, method='emd', threshold=0.05)
    print(res_emd.report())
    
    # 6. Visual Verification: Distribution Plots (topo_density)
    # 6. Visual Verification: Documentation Composite (Match vs Mismatch)
    print("\n--- [Visual] Generating Documentation Composite (Match vs Mismatch) ---")
    
    # Create a custom figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    # LEFT: Objective Space (Match) - Plotting f1 (Axis 0)
    # We use a trick: call topo_density with a specific axis and pass our custom axes object
    # Note: topo_density usually creates its own fig, but we can reuse the logic if we extract it 
    # or just use it independently.
    # Since topo_density is high-level, let's call it for independent plots and then save/show.
    # Actually, simpler: just generate the two independent figures as before, but label them clearly 
    # so the user knows which is which.
    
    # Plot 1: Objective Space Match (Axis 0 only for clarity)
    mb.view.topo_density(exp1, exp2, space='objs', axes=[0], alpha=0.01, ax=ax1,
                         title="LEFT: Objective Match (Convergence)")
                         
    # Plot 2: Decision Space Mismatch (Axis 2 has strong divergence)
    # Based on previous run, Axis 3-5 were divergent. Let's pick Axis 2 (index 2)
    mb.view.topo_density(exp1, exp2, space='vars', axes=[2], alpha=0.05, ax=ax2,
                         title="RIGHT: Decision Mismatch (Strategy)")
    
    # Set explicit titles on axes since topo_density might skip it in external mode or we want to override
    ax1.set_title("Objective Match (f1)", fontsize=11, fontweight='bold')
    ax2.set_title("Decision Mismatch (x3)", fontsize=11, fontweight='bold')

    print("Check the generated figures: Figure 1 is Match, Figure 2 is Mismatch.")
    
    # Save for documentation (Overwrite the standard figure with this composite)
    # Ensure the directory exists or just save to relative path if running from root
    import os
    save_path = "docs/images/topo_density.png"
    if os.path.exists("docs/images"):
        fig.savefig(save_path, dpi=100)
        print(f"Composite figure saved as '{save_path}' (Documentation Updated)")
    else:
        fig.savefig("topo_density_composite.png", dpi=100)
        print("Composite figure saved as 'topo_density_composite.png' (Docs folder not found)")

    # Show the interactive window (fixes regression reported by user)
    plt.show()

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# 1. Objective Space Match: The overlapping curves in the first plot confirm
#    that both algorithms cover the same performance region.
#
# 2. Decision Space Divergence: The separated curves in the second plot 
#    reveal that they achieve this performance using different variable ranges.

