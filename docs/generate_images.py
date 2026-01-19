import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

from MoeaBench import mb
from MoeaBench.view.style import apply_style, OCEAN_PALETTE

# Mock plt.show to prevent it from clearing the figure
plt.show = lambda: None
apply_style()

def generate():
    img_dir = "docs/images"
    os.makedirs(img_dir, exist_ok=True)
    
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
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.SPEA2(population=100, generations=30)
    exp2.run(repeat=5)
    mb.view.tierplot(exp, exp2, gen=10)
    plt.savefig(os.path.join(img_dir, "tierplot.png"), bbox_inches='tight', dpi=150)
    plt.close('all')
    
    print("\nAll assets generated successfully (Single vs. Multi-run distinction applied).")

if __name__ == "__main__":
    generate()
