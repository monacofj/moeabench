import os
import sys
import numpy as np
import re

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

def parse_filename(filename):
    """
    Extracts MOP name and parameters from the Delgado2 filename.
    Format example: legacy_DTLZ1_M_3_K_5_N_7_samples_1000.csv
    """
    mop_match = re.search(r"legacy_(DTLZ\d+|DPF\d+)", filename)
    if not mop_match:
        return None
    
    mop_name = mop_match.group(1)
    
    params = {}
    for p in ['M', 'K', 'N', 'D']:
        match = re.search(rf"_{p}_(\d+)", filename)
        if match:
            params[p] = int(match.group(1))
            
    return mop_name, params

def audit_legacy2_optimal():
    data_dir = os.path.join(PROJECT_ROOT, "tests/legacy2_optimal")
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory not found: {data_dir}")
        return

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    csv_files.sort()

    print(f"{'Problem':<10} | {'M':<2} | {'N':<3} | {'Inv Type':<10} | {'Legacy Mean':<12} | {'Status'}")
    print("-" * 75)

    for f in csv_files:
        parsed = parse_filename(f)
        if not parsed:
            continue
            
        mop_name, params = parsed
        M = params.get('M', 3)
        
        # Load Legacy2_optimal Data using numpy
        file_path = os.path.join(data_dir, f)
        try:
            F = np.loadtxt(file_path, delimiter=",")
        except Exception as e:
            print(f"{mop_name:<10} | Error reading CSV: {e}")
            continue
        
        # Define Invariants
        label = "N/A"
        gap = 0
        status = "UNKNOWN"

        if "DTLZ1" in mop_name:
            vals = np.sum(F, axis=1)
            target = 0.5
            label = "Sum 0.5"
            gap = np.mean(np.abs(vals - target))
        elif any(x in mop_name for x in ["DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6"]):
            vals = np.sum(F**2, axis=1)
            target = 1.0
            label = "SOS 1.0"
            gap = np.mean(np.abs(vals - target))
        elif "DTLZ8" in mop_name:
            # fM + 4fi = 1.0
            vals = F[:, -1] + 4 * F[:, 0]
            target = 1.0
            label = "Line 1.0"
            gap = np.mean(np.abs(vals - target))
        elif "DTLZ9" in mop_name:
            # fj^2 + fM^2 = 1.0
            vals = F[:, 0]**2 + F[:, -1]**2
            target = 1.0
            label = "SOS_pair 1.0"
            gap = np.mean(np.abs(vals - target))
        elif any(x in mop_name for x in ["DPF", "DTLZ7"]):
            label = "SKIP"
            gap = 0
        else:
            label = "SKIP"
            gap = 0

        if label != "SKIP":
            status = "OK" if gap < 1e-4 else f"FAIL ({gap:.2e})"
            print(f"{mop_name:<10} | {M:<2} | {params.get('N'):<3} | {label:<10} | {np.mean(vals):<12.6f} | {status}")
        else:
            print(f"{mop_name:<10} | {M:<2} | {params.get('N'):<3} | {label:<10} | {'-':<12} | MANUAL")

if __name__ == "__main__":
    audit_legacy2_optimal()
