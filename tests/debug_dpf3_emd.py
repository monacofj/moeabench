
import numpy as np
import MoeaBench as mb
from MoeaBench.mops import DPF3
from MoeaBench.metrics import emd

def debug_dpf3():
    print("--- Debugging DPF3 EMD Anomaly ---")
    
    # 1. Instantiate DPF3
    problem = DPF3()
    print(f"Problem: {problem.__class__.__name__} (N={problem.N}, M={problem.M})")
    
    # 2. Get Reference Front (GT)
    gt = problem.pf(n_points=500)
    print(f"GT Shape: {gt.shape}")
    print(f"GT Range: {np.min(gt, axis=0)} to {np.max(gt, axis=0)}")
    
    # 3. Simulate a MOEA/D-like population (e.g. slight noise around GT)
    # The anomaly is that MOEA/D had EMD=0.0000. This implies perfect match?
    # Let's test EMD(GT, GT) first.
    
    val_self = emd(gt, ref=gt)
    print(f"\nTest 1: EMD(GT, GT) -> {val_self.values[0][0]}")
    
    # 4. Simulate small noise
    noisy = gt + np.random.normal(0, 0.00001, gt.shape)
    val_noise = emd(noisy, ref=gt)
    print(f"Test 2: EMD(GT+epsilon, GT) -> {val_noise.values[0][0]}")

    # DEBUG ANALYSIS
    # Hypothesis: DPF3 collapses to corners due to x^100
    from MoeaBench.metrics.evaluator import get_reference_front, _extract_data
    
    _, Fs, _, _ = _extract_data(gt)
    ref_front = get_reference_front(gt, Fs) # This performs NDS
    
    print(f"\nAnalysis:")
    print(f"Original GT Size: {len(gt)}")
    print(f"Ref Front Size (Non-Dominated): {len(ref_front)}")
    
    print("\nSample Points from Ref Front:")
    print(ref_front[:5])
    
    # Check discreteness
    unique_vals = np.unique(np.round(ref_front, 6), axis=0)
    print(f"\nUnique Points (rounded 6 decimals): {len(unique_vals)}")
    
    import matplotlib.pyplot as plt
    # No plotting, just data analysis
    
if __name__ == "__main__":
    debug_dpf3()
