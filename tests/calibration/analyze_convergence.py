
import os
import pandas as pd
import numpy as np
import sys

# Ensure local MoeaBench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from pymoo.indicators.igd import IGD

DATA_DIR = os.path.join(PROJ_ROOT, "tests/calibration_data")
GT_DIR = os.path.join(PROJ_ROOT, "tests/ground_truth")

def analyze():
    mops = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7", "DTLZ8", "DTLZ9", 
            "DPF1", "DPF2", "DPF3", "DPF4", "DPF5"]
    algs = ["NSGA2", "NSGA3", "MOEAD"]
    
    results = []

    for mop in mops:
        gt_file = os.path.join(GT_DIR, f"{mop}_3_optimal.csv")
        if not os.path.exists(gt_file):
            continue
        
        F_opt = pd.read_csv(gt_file, header=None).values
        # Pymoo IGD object
        metric = IGD(F_opt, zero_to_one=True)
        
        for alg in algs:
            # Check for snapshots
            snapshots = []
            for gen in range(100, 1001, 100):
                file = os.path.join(DATA_DIR, f"{mop}_{alg}_standard_run00_gen{gen}.csv")
                if os.path.exists(file):
                    snapshots.append((gen, file))
            
            # Add final result (gen 1000) if not already there
            final_file = os.path.join(DATA_DIR, f"{mop}_{alg}_standard_run00.csv")
            if os.path.exists(final_file) and not any(s[0] == 1000 for s in snapshots):
                snapshots.append((1000, final_file))
                
            if not snapshots:
                continue
                
            # Calculate IGD for each snapshot
            igds = []
            for gen, fpath in sorted(snapshots):
                F = pd.read_csv(fpath).values
                # Ensure 3 objectives
                if F.shape[1] > 3: F = F[:, :3]
                
                val = float(metric.do(F))
                igds.append((gen, val))
            
            # Find convergence point
            # Definition: First gen G where IGD(G) <= 1.05 * Final_IGD
            final_igd = igds[-1][1]
            t_conv = 1000
            for gen, val in igds:
                if val <= 1.05 * final_igd: # Using 5% margin for stability
                    t_conv = gen
                    break
            
            results.append({
                "MOP": mop,
                "Algorithm": alg,
                "Final_IGD": final_igd,
                "T_conv": t_conv
            })

    df = pd.DataFrame(results)
    # Filter out DTLZ8/9 or MOEAD where it skipped
    df = df[df['Final_IGD'] < 100] # Ignore failures stuck at 1000 (like DTLZ9)
    
    pivot = df.pivot(index="MOP", columns="Algorithm", values="T_conv")
    print("\n--- Generations to Stabilization (T-conv) ---")
    print(pivot.to_string())
    
    # Save for reference
    df.to_csv(os.path.join(PROJ_ROOT, "tests/convergence_analysis.csv"), index=False)

if __name__ == "__main__":
    analyze()
