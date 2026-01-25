import os
import sys
import site

# --- Environment Fix for Matplotlib 3D ---
try:
    import mpl_toolkits
    user_site = site.getusersitepackages()
    local_mpl = os.path.join(user_site, 'mpl_toolkits')
    if os.path.exists(local_mpl) and hasattr(mpl_toolkits, '__path__'):
        if local_mpl not in mpl_toolkits.__path__:
            mpl_toolkits.__path__.insert(0, local_mpl)
    from mpl_toolkits import mplot3d
except:
    pass
# -----------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Discover Root
def discover_root():
    curr = os.path.dirname(os.path.abspath(__file__))
    for _ in range(3):
        if os.path.exists(os.path.join(curr, "MoeaBench")):
            return curr
        curr = os.path.dirname(curr)
    return os.getcwd()

PROJECT_ROOT = discover_root()
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import MoeaBench as mb
from MoeaBench.core import SmartArray

# Monkeypatch plt.show
plt.show = lambda: None

def generate_fig(mop_name, M=3):
    print(f"Generating benchmark figure for: {mop_name}...")
    exp = mb.experiment()
    try:
        mop = getattr(mb.mops, mop_name)(M=M)
    except:
        mop = getattr(mb.mops, mop_name)(M=M, D=M-1)
    exp.mop = mop
    
    # 1. New v0.7.5 Ground Truth
    try:
        f_gt_pop = exp.optimal(n_points=1000)
        F_gt = f_gt_pop.objectives
    except Exception:
        # Fallback to frozen CSV
        gt_file = os.path.join(PROJECT_ROOT, "tests/ground_truth", f"{mop_name}_{M}_optimal.csv")
        if os.path.exists(gt_file):
            F_gt = pd.read_csv(gt_file, header=None).values
        else:
            print(f"Skipping {mop_name}: No truth data found.")
            return

    gt_series = SmartArray(F_gt, name="v0.7.5 GT")
    
    # 2. Legacy Audit Reference
    prefix = "lg__" if "DTLZ" in mop_name else "lg_"
    # We prioritize 'front.csv' if it exists, otherwise 'opt_front.csv'
    leg_path = os.path.join(PROJECT_ROOT, "tests/audit_data", f"legacy_{mop_name}", f"{prefix}{mop_name}_{M}_front.csv")
    if not os.path.exists(leg_path):
        leg_path = os.path.join(PROJECT_ROOT, "tests/audit_data", f"legacy_{mop_name}", f"{prefix}{mop_name}_{M}_opt_front.csv")
    
    leg_series = None
    if os.path.exists(leg_path):
        F_leg = pd.read_csv(leg_path, header=None).values
        leg_series = SmartArray(F_leg, name="Audit Legacy")

    # 3. Plot
    plt.figure(figsize=(10, 8))
    if leg_series is not None:
        mb.view.topo_shape(gt_series, leg_series, title=f"{mop_name} (M={M})", mode='static')
    else:
        mb.view.topo_shape(gt_series, title=f"{mop_name} (M={M})", mode='static')
    
    save_path = os.path.join(PROJECT_ROOT, "misc/figs", f"{mop_name}_audit.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    benchmarks = [f"DTLZ{i}" for i in range(1, 10)] + [f"DPF{i}" for i in range(1, 6)]
    for b in benchmarks:
        generate_fig(b)
