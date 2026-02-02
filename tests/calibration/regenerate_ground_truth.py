
import os
import sys
import pandas as pd
import numpy as np

# Ensure local MoeaBench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import MoeaBench as mb

GT_DIR = os.path.join(PROJ_ROOT, "tests/ground_truth")
os.makedirs(GT_DIR, exist_ok=True)

def regenerate_ground_truth():
    print("=== Regenerating Static Ground Truth Files ===")
    print(f"Target Directory: {GT_DIR}")
    
    # We only care about M=3 for the certification report visualizer
    # But checking the directory listing, there are files for 3, 5, 10, 31 objectives.
    # We should probably update at least the M=3 ones which are used for the report.
    # Let's target what is currently there.
    
    # List of MOPs typically used
    mops = [f"DTLZ{i}" for i in range(1, 10)] + [f"DPF{i}" for i in range(1, 6)]
    
    # Objectives found in directory listing
    # {"name":"DPF1_10_optimal.csv".. "DPF1_3_optimal.csv"...}
    # It seems baseline generates for 3, 5, 10, 31?
    # For certification visual report we strictly use M=3.
    # Let's update M=3 first and foremost.
    
    m_values = [3] 
    
    for mop_name in mops:
        for m in m_values:
            print(f"Processing {mop_name} (M={m})... ", end="")
            
            try:
                # Instantiate MOP
                # Handle DPF/DTLZ7 parameter requirements if any
                try:
                    mop_obj = getattr(mb.mops, mop_name)(M=m)
                except TypeError:
                    # Some MOPs like DPF might need D explicit if M is low?
                    # Or maybe D defaults to M-1?
                    # In compute_baselines checking logic:
                    mop_obj = getattr(mb.mops, mop_name)(M=m, D=m-1)
            
                # Determine density
                # DPF curves (1D) need higher density to avoid "HV > 100%" artifact
                # DTLZ surfaces (2D) are fine with 2k usually, but let's stick to 2k for now unless asked.
                if "DPF" in mop_name:
                    n_points = 10000 
                else:
                    n_points = 2000
                
                # Generate
                exp = mb.experiment(mop=mop_obj)
                F_opt = exp.optimal(n_points=n_points).objs
                
                # Save
                filename = f"{mop_name}_{m}_optimal.csv"
                filepath = os.path.join(GT_DIR, filename)
                
                pd.DataFrame(F_opt).to_csv(filepath, index=False, header=False)
                print(f"Done. ({n_points} points) -> {filename}")
                
            except Exception as e:
                print(f"FAILED: {str(e)}")

    print("=== Regeneration Complete ===")

if __name__ == "__main__":
    regenerate_ground_truth()
