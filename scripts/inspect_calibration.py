
import os
import sys

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import pandas as pd
import numpy as np
from MoeaBench.clinic import igd_efficiency, emd_efficiency, get_floor
from MoeaBench.diagnostics.enums import DiagnosticProfile

BASELINE_FILE = os.path.join(PROJ_ROOT, "tests/baselines_v0.8.0.csv")

def inspect():
    df = pd.read_csv(BASELINE_FILE)
    df = df[df['Intensity'] == 'standard']
    
    print(f"{'MOP':<8} | {'ALG':<8} | {'IGD Raw':<10} | {'Floor':<10} | {'Eff Ratio':<10} | {'STD Verdict':<15}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        mop = row['MOP']
        alg = row['Algorithm']
        igd = row['IGD_mean']
        
        floor = get_floor(mop, "igd_floor", 200, "p10")
        eff = igd / floor if floor > 1e-9 else 0
        
        verdict = "PASS" if eff <= 1.6 else "FAIL"
        if eff <= 1.1: verdict += " (RES)"

        emd_floor = get_floor(mop, "emd_floor", 200, "p10")
        emd_eff = 0.0
        # EMD value might need to be fetched from df or recomputed?
        # The df has 'EMD_mean' ? Let's check columns used.
        # Ah, inspect() iterates df rows. Row has 'EMD_mean' ?
        # The script does: igd = row['IGD_mean'].
        # Let's add emd = row['EMD_val'] or similar. 
        # Wait, the DF column name in generating report was not saved to baseline csv?
        # Report generator saves summary to `tests/calibration_data/...` but baseline csv is summary of 30 runs?
        # generate_visual_report reads BASELINE_FILE.
        # Let's check BASELINE_FILE columns.
        
        emd_raw = row.get('EMD_mean', 0.0) 
        sp_raw = row.get('SP_mean', 0.0)
        
        if emd_floor > 1e-9:
             emd_eff = emd_raw / emd_floor
        
        print(f"{mop:<8} | {alg:<8} | IGD:{eff:.1f}x | SP:{sp_raw:.4f} | {verdict:<15}")

if __name__ == "__main__":
    inspect()
