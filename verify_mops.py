# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

#!/usr/bin/env python3
import numpy as np
from MoeaBench import mb

def verify_mop(name, **kwargs):
    print(f"\n--- Verifying {name} ---")
    try:
        # 1. Factory access
        factory = getattr(mb.mops, name)(**kwargs)
        print(f"Factory {name} created.")
        
        # 2. Specialization
        inst = factory()
        print(f"Instance specialized: {type(inst).__name__}")
        
        # 3. Metadata check
        M = getattr(inst, 'M', 3)
        K = getattr(inst, 'K', 5)
        
        # Robust N detection
        N = getattr(inst, 'N', None)
        if N is None:
            if hasattr(inst, 'get_Nvar'):
                N = inst.get_Nvar()
            elif hasattr(inst, 'get_CACHE'):
                N = inst.get_CACHE().get_BENCH_CI().get_Nvar()
            elif M and K:
                N = M + K - 1
        
        print(f"Metadata: M={M}, N={N}, K={K}")
        
        # 4. Evaluation test
        if N:
            X = np.random.random((10, N))
            res = inst.evaluation(X, 0)
            print(f"Evaluation F shape: {res['F'].shape}")
            if 'G' in res:
                print(f"Evaluation G shape: {res['G'].shape}")
            
            # Check some values (not NaN)
            if np.any(np.isnan(res['F'])):
                print("WARNING: NaN values detected in F!")
            else:
                print("Evaluation F looks healthy (no NaNs).")
        else:
            print("ERROR: Could not determine Nvar.")
            
    except Exception as e:
        print(f"ERROR during {name} verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # DTLZ Family
    for i in range(1, 10):
        verify_mop(f"DTLZ{i}")
        
    # DPF Family
    for i in range(1, 6):
        verify_mop(f"DPF{i}", D=2)
