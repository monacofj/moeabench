
import os
import sys
import numpy as np
import pandas as pd
import warnings

# Ensure local MoeaBench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from MoeaBench.mops.DPF4 import DPF4

def main():
    print("--- DPF4 Alignment Audit ---")
    
    # 1. Deterministic Ground Truth Generation (g=0)
    print("\n1. Generating Theoretical Ground Truth (Codebase Logic)...")
    try:
        problem = DPF4(M=3, D=3) # Standard DPF4 config
        
        # To get g=0 in DPF4 (Rastrigin-like), x_m should be 0.5
        # The variables associated with 'g' are from D onwards.
        # But wait, D=3 means x[0], x[1] are position, x[2:] are distance?
        # Let's check the class.
        # D is usually number of objectives in DTLZ, but in DPF?
        # In DPF4.py: D, K, M = self.D, self.K, self.M
        # X_m = X[:, D-1:]
        # So x[0]...x[D-2] are position?
        
        # Let's look at BaseDPF to be sure about K and D relation.
        # DPF4 default: M=3.
        # If D is not passed, what is it?
        # This script assumes we can inspect the instance.
        
        print(f"  Target: M={problem.M}, D={problem.D}, K={problem.K}")
        
        # Generate PF points:
        # We need samples where X_m = 0.5 (to make g=0)
        # And vary X_p (x[0]...x[D-2])
        
        n_samples = 100
        X_p = np.random.rand(n_samples, problem.D - 1) # Unsafe random just for check? No, let's span.
        X_p = np.linspace(0, 1, n_samples).reshape(-1, 1) # Simplify for 1 freq variable (D=2?)
        # Wait, if D=3, we have 2 position variables?
        # No, D is usually objectives count in standard definition, but here D is likely "k" related?
        # Let's blindly trust the problem.N based on BaseMop logic
        
        n_var = problem.N
        X_test = np.full((10, n_var), 0.5)
        # Set g-vars (from D-1 onwards) to 0.5 -> g=0
        # Vary the first variable
        X_test[:, 0] = np.linspace(0, 1, 10)
        
        res = problem.evaluation(X_test)
        F_code = res['F']
        
        print(f"  Generated {len(F_code)} points with forced g=0.")
        print(f"  Sample F[0]: {F_code[0]}")
        
    except Exception as e:
        print(f"FAIL: Could not generate code GT: {e}")
        return

    # 2. Compare against Frozen Ground Truth
    gt_file = os.path.join(PROJ_ROOT, "tests/ground_truth/DPF4_3_optimal.csv")
    print(f"\n2. Checking Frozen Ground Truth: {gt_file}")
    if os.path.exists(gt_file):
        F_frozen = pd.read_csv(gt_file, header=None).values
        ideal = np.min(F_frozen, axis=0)
        nadir = np.max(F_frozen, axis=0)
        print(f"  Frozen Ideal: {ideal}")
        print(f"  Frozen Nadir: {nadir}")
        
        # Check if F_code roughly fits in F_frozen range
        # (This is a coarse check)
        in_range = np.all((F_code >= ideal - 0.1) & (F_code <= nadir + 0.1))
        print(f"  Code-generated (g=0) points fall within Frozen bounds? {in_range}")
        
        if not in_range:
             print("  ALERT: Code generation produces points outside frozen GT bounds!")
             print(f"  Code (g=0) Range: {np.min(F_code, axis=0)} to {np.max(F_code, axis=0)}")
    else:
        print("  FAIL: Ground Truth file not found.")

    # 3. Analyze Final Populations ($g$-value check)
    print("\n3. Analyzing MOEA Final Populations for Trap Depth...")
    data_dir = os.path.join(PROJ_ROOT, "tests/calibration_data")
    
    # Check NSGA2/NSGA3/MOEAD
    for alg in ["NSGA2", "NSGA3", "MOEAD"]:
        pop_file = f"DPF4_{alg}_standard_run00_X.csv" # Do we have X? 
        # Wait, the calibration runner saves F, but does it save X?
        # generate_baselines.py: saves *_gen{g}.csv (F only usually)
        # Let's check if we have standard_run00.csv (F) and if we have X.
        # The prompt said "Implement persistence for Shadow Data (F & X)".
        # Let's check one file.
        
        # Assuming we might only have F in the CSVs used for report.
        # If we don't have X, we can't calc g.
        # Let's check file existence first.
        pass

    # Actually, let's verify if X files exist in calibration_data
    # If not, we cannot calculate g.
    
    # Search for X files
    found_X = False
    for f in os.listdir(data_dir):
        if "DPF4" in f and "_X.csv" in f:
             # Found one!
             print(f"  Found design space file: {f}")
             X_pop = pd.read_csv(os.path.join(data_dir, f)).values
             
             # Calculate g manually
             # DPF4 logic:
             D = problem.D
             X_m = X_pop[:, D-1:]
             k = X_m.shape[1]
             # Rastrigin g
             g_vals = 100 * (k + np.sum((X_m - 0.5)**2 - np.cos(20 * np.pi * (X_m - 0.5)), axis=1))
             
             print(f"  --> Mean g-value: {np.mean(g_vals):.4f} (Target 0.0)")
             print(f"  --> Min g-value:  {np.min(g_vals):.4f}")
             print(f"  --> Max g-value:  {np.max(g_vals):.4f}")
             
             if np.mean(g_vals) > 1.0:
                 print("  CONCLUSION: Algorithm trapped in local optima (g >> 0).")
             else:
                 print("  CONCLUSION: Algorithm converged to g~0. Optimization successful?")
             found_X = True
             
    if not found_X:
        print("  WARNING: No X (Design Space) files found for DPF4. Cannot verify g-values directly.")
        print("  (Did generate_baselines.py save X? Need to check.)")

if __name__ == "__main__":
    main()
