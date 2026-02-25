import mb_path
from MoeaBench import mb
from MoeaBench.diagnostics import baselines

def check_baselines():
    problem = "DTLZ1"
    k = 200
    
    # Check CLOSENESS baseline
    # get_baseline_values returns (ideal, rand50)
    ideal, rand50 = baselines.get_baseline_values(problem, k, "closeness")
    
    # Check HEADWAY baseline (to see difference)
    _, hw_rand50 = baselines.get_baseline_values(problem, k, "headway")
    
    print(f"Problem: {problem}, K: {k}")
    print(f"Closeness Baseline (rand50): {rand50}")
    print(f"Headway Baseline (rand50): {hw_rand50}")
    
    # Calculate s_k from mop
    mop = mb.mops.DTLZ1(M=3)
    s_k = getattr(mop, 's_fit', getattr(mop, 's_k', 1.0))
    print(f"Default s_k from MOP: {s_k}")

if __name__ == "__main__":
    check_baselines()
