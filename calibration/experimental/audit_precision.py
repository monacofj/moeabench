# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Audit: Precision Integrity (Topic D)
====================================

Verifies that the calibration pipeline maintains 64-bit floating point precision (Float64).
Audits:
1. Data Ingestion (pd.read_csv dtypes).
2. Metric Internals (evaluator.normalize).

Usage:
    python3 tests/calibration/audit_precision.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure local MoeaBench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

DATA_DIR = os.path.join(PROJ_ROOT, "tests/calibration_data")

def audit_precision():
    print("--- Precision Audit: Float64 Verification ---")
    
    # 1. Audit Data Ingestion
    print("\n[1] Checking Calibration Data Types...")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f != "generation_stats.csv"]
    
    # Sample a few files
    sample_files = files[:5] 
    all_pass = True
    
    for f in sample_files:
        path = os.path.join(DATA_DIR, f)
        df = pd.read_csv(path)
        
        # Check objectives (usually first few columns)
        # Assuming first 3 columns are objectives
        objs = df.iloc[:, 0:min(3, df.shape[1])]
        dtypes = objs.dtypes
        
        print(f"  File: {f}")
        for col, dtype in dtypes.items():
            is_64 = (dtype == np.float64)
            status = "PASS" if is_64 else "FAIL"
            print(f"    Col {col}: {dtype} -> {status}")
            if not is_64:
                all_pass = False
                
    if all_pass:
        print(">> Ingestion: ALL CHECKS PASSED. Data loaded as Float64.")
    else:
        print(">> Ingestion: FAIL. Some data loaded as lower precision.")

    # 2. Audit Metric Internals (Simulated)
    print("\n[2] Verifying Metric Internal Precision...")
    from MoeaBench.metrics.evaluator import normalize
    
    # Create float64 arrays
    A = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    B = [np.array([[0.5, 0.6]], dtype=np.float64)]
    
    min_val, max_val = normalize([A], B)
    print(f"  Input Type: {A.dtype}")
    print(f"  Output Min Type: {min_val.dtype}")
    print(f"  Output Max Type: {max_val.dtype}")
    
    if min_val.dtype == np.float64 and max_val.dtype == np.float64:
        print(">> Normalization: PASS. Maintained Float64.")
    else:
        print(">> Normalization: FAIL. Downcasted.")

if __name__ == "__main__":
    audit_precision()
