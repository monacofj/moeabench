# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import pandas as pd
import numpy as np

# Ensure local moeabench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import moeabench as mb

DATA_DIR = os.path.join(PROJ_ROOT, "calibration/data/calibration_data")

def generate():
    mops = ["DTLZ8", "DTLZ9"]
    alg = "MOEAD"
    # Intensities as per compute_baselines.py and list_dir analysis
    intensities = {
        "light": {"pop": 52, "gen": 100, "runs": 5},
        "standard": {"pop": 200, "gen": 1000, "runs": 5}
    }
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for mop_name in mops:
        mop = getattr(mb.mops, mop_name)(M=3)
        for int_name, params in intensities.items():
            pop_size = params["pop"]
            n_gen = params["gen"]
            n_runs = params["runs"]
            
            print(f"Generating {mop_name} | {alg} | {int_name} ({n_runs} runs)...")
            
            for i in range(n_runs):
                seed = 42 + i
                moea = mb.moeas.MOEAD(population=pop_size, generations=n_gen, seed=seed)
                exp = mb.experiment(mop=mop, moea=moea)
                exp.run()
                
                # Final population
                res = exp[0]
                objs = res.pop().objs
                df = pd.DataFrame(objs, columns=[f"f{j+1}" for j in range(objs.shape[1])])
                filename = f"{mop_name}_{alg}_{int_name}_run{i:02d}.csv"
                df.to_csv(os.path.join(DATA_DIR, filename), index=False)
                
                # Snapshots for run00 standard (for convergence analysis)
                if i == 0 and int_name == "standard":
                    print(f"  Saving snapshots for run00 standard...")
                    try:
                        f_hist = res.history('f')
                        for g in range(100, n_gen + 1, 100):
                            if g <= len(f_hist):
                                # history is 0-indexed, so gen 100 is at index 99
                                F_g = f_hist[g-1]
                                df_g = pd.DataFrame(F_g, columns=[f"f{j+1}" for j in range(F_g.shape[1])])
                                snap_filename = f"{mop_name}_{alg}_{int_name}_run{i:02d}_gen{g}.csv"
                                df_g.to_csv(os.path.join(DATA_DIR, snap_filename), index=False)
                    except Exception as e:
                        print(f"  Warning: Could not save snapshots: {e}")
                
                print(f"  Saved {filename}")

if __name__ == "__main__":
    generate()
