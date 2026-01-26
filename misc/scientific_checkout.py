import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure local MoeaBench is importable
sys.path.append(os.path.abspath("."))
import MoeaBench as mb

def run_scientific_checkout():
    print(f"=== MoeaBench v{mb.system.version()} Scientific Checkout PoC ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    # 1. Configuration
    mops_to_test = ["DTLZ2", "DTLZ4", "DTLZ8", "DPF2"]
    algs_to_test = ["NSGA2", "NSGA3", "MOEAD"]
    M = 3
    n_runs = 5
    n_gen = 200
    pop_size = 100

    results = []

    for mop_name in mops_to_test:
        print(f"\n[Problem] {mop_name} (M={M})")
        
        # Instantiate MOP
        try:
            mop = getattr(mb.mops, mop_name)(M=M)
        except:
            # Handle DPF which requires D
            mop = getattr(mb.mops, mop_name)(M=M, D=2)
            
        # Get Analytical Optimal (Ground Truth)
        opt_pop = mb.experiment(mop=mop).optimal(n_points=1000)
        F_opt = opt_pop.objectives

        for alg_name in algs_to_test:
            # Skip MOEA/D for DTLZ8 (Constraint matching issues in Pymoo)
            if alg_name == "MOEAD" and mop_name == "DTLZ8":
                print(f"  > Skipping {alg_name} for {mop_name} (No constraint support)")
                continue

            print(f"  > Running {alg_name}...", end="", flush=True)
            
            exp = mb.experiment()
            exp.mop = mop
            
            # Setup Algorithm
            if alg_name == "NSGA2":
                alg = mb.moeas.NSGA2()
            elif alg_name == "NSGA3":
                alg = mb.moeas.NSGA3()
            elif alg_name == "MOEAD":
                alg = mb.moeas.MOEAD()
            
            exp.moea = alg
            
            # Run Experiment
            # Note: repeat=n_runs, and kwargs like 'generations' and 'population' 
            # are propagated to the MOEA instance.
            exp.run(repeat=n_runs, generations=n_gen, population=pop_size)
            
            # 1. Metric: IGD (Proximity)
            # MetricMatrix returns a (G, R) grid. We want the mean of the last generation.
            igd_val = mb.metrics.igd(exp, F_opt).gens(-1).mean()
            
            # 2. Metric: KS-test (Topology/Distribution)
            # Use topo_dist to compare the final non-dominated front with GT.
            # It returns a DistMatchResult with per-axis p-values.
            combined_front = exp.pop().objectives
            topo_match = mb.stats.topo_distribution(combined_front, F_opt)
            p_vals = topo_match.p_values # Dictionary of {axis: p_val}
            ks_p_val = np.mean(list(p_vals.values()))
            
            # 3. Metric: A12 (Probability)
            # For checkout, we can compare this alg's HV vs a nominal high value
            # But the Rule 10 trio usually looks for relative performance.
            # Here, we'll just log IGD and KS as the primary checkout triggers.
            
            print(f" Done. [IGD={igd_val:.4f}, KS_p={ks_p_val:.4f}]")
            
            results.append({
                "MOP": mop_name,
                "Algorithm": alg_name,
                "IGD_mean": igd_val,
                "KS_p_val": ks_p_val
            })

    # Summary Table
    print("\n" + "="*60)
    print(f"{'Problem':<10} | {'Algorithm':<10} | {'IGD':<8} | {'KS p-val':<8}")
    print("-" * 60)
    for res in results:
        status = "OK" if res["IGD_mean"] < 0.05 else "WARN"
        print(f"{res['MOP']:<10} | {res['Algorithm']:<10} | {res['IGD_mean']:<8.4f} | {res['KS_p_val']:<8.4f} | {status}")
    print("="*60)

if __name__ == "__main__":
    run_scientific_checkout()
