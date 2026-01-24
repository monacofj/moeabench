
import os
import numpy as np
import pandas as pd
import MoeaBench as mb
from MoeaBench.core.run import Population

# Ensure determinism
np.random.seed(42)

def generate_analytical_truth(mop_name, M, out_dir):
    """Generates the perfect analytical Pareto Front."""
    print(f"  - Generating analytical front for {mop_name} (M={M})...")
    mop_cls = getattr(mb.mops, mop_name)
    
    # N logic to ensure scale invariance is tested
    N_val = max(M + 20, 31)
    
    if mop_name.startswith("DPF"):
        # Use D=2 as the base for degenerate fronts to keep them challenging
        mop = mop_cls(M=M, D=2, K=10, N=N_val)
    else:
        mop = mop_cls(M=M, N=N_val)
        
    exp = mb.experiment(mop)
    # n_points: 100 is enough for a dense curve/surface in CSV
    opt_pop = exp.optimal(n_points=500)
    F = np.asarray(opt_pop.objectives)
    
    filename = f"{mop_name}_{M}_optimal.csv"
    filepath = os.path.join(out_dir, filename)
    pd.DataFrame(F).to_csv(filepath, index=False, header=False)
    return filepath

def generate_dtlz8_heuristic_truth(M, out_dir):
    """
    Generates a high-fidelity proxy for DTLZ8 using NSGA-III.
    Uses multi-seed aggregation for maximum coverage.
    """
    print(f"  - Generating heuristic front for DTLZ8 (M={M}) via NSGA-III (Multi-Seed)...")
    mop = mb.mops.DTLZ8(M=M)
    
    all_fronts = []
    seeds = [1, 2, 3, 4, 5]
    
    # High fidelity setup
    pop_size = 200 if M < 10 else 400
    generations = 500
    
    for seed in seeds:
        print(f"    - Running seed {seed}...")
        moea = mb.moeas.NSGA3(seed=seed)
        exp = mb.experiment(mop, moea)
        exp.run(pop_size=pop_size, generations=generations)
        run = exp.last_run
        all_fronts.append(run.history('nd')[-1]) # Last generation non-dominated
        
    # Aggregate and filter
    combined_F = np.vstack(all_fronts)
    # We use mb.core.Population.non_dominated() to filter the combined results
    dummy_X = np.zeros((combined_F.shape[0], mop.N))
    agg_pop = Population(combined_F, dummy_X).non_dominated()
    F_refined = np.asarray(agg_pop.objectives)
    
    filename = f"DTLZ8_{M}_optimal.csv" # We name it optimal even if heuristic for framework consistency
    filepath = os.path.join(out_dir, filename)
    pd.DataFrame(F_refined).to_csv(filepath, index=False, header=False)
    return filepath

def main():
    out_dir = "tests/ground_truth"
    os.makedirs(out_dir, exist_ok=True)
    
    analytical_mops = [
        "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7", "DTLZ9",
        "DPF1", "DPF2", "DPF3", "DPF4", "DPF5"
    ]
    
    # Target dimensions
    dimensions = [3, 5, 10, 31]
    
    print(f"Creating Ground Truth in {out_dir}...")
    
    for M in dimensions:
        print(f"\nProcessing M={M}:")
        for mop in analytical_mops:
            generate_analytical_truth(mop, M, out_dir)
        
        # DTLZ8 Heuristic
        generate_dtlz8_heuristic_truth(M, out_dir)

    print("\nGround Truth Generation Complete.")

if __name__ == "__main__":
    main()
