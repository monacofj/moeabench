
import MoeaBench as mb
import numpy as np

def debug_dtlz1():
    mop = mb.mops.DTLZ1(M=3, N=7) # Standard N=7
    moea = mb.moeas.NSGA3(seed=42)
    exp = mb.experiment(mop, moea)
    
    print("Running DTLZ1 M=3 N=7 (NSGA3, Pop 150, Gen 300)...")
    exp.run(pop_size=150, generations=300)
    
    f_final = np.asarray(exp.last_run.front())
    print(f"Number of points in front: {len(f_final)}")
    
    sums = np.sum(f_final, axis=1)
    print(f"Sum of objectives (Target 0.5):")
    print(f"  Min: {np.min(sums):.4f}")
    print(f"  Max: {np.max(sums):.4f}")
    print(f"  Mean: {np.mean(sums):.4f}")
    
    if np.mean(sums) > 1.0:
        print("FAIL: Did not converge to the Pareto front.")
    else:
        print("SUCCESS: Converged (approx).")

if __name__ == "__main__":
    debug_dtlz1()
