
from MoeaBench import mb
import numpy as np

def verify():
    print("--- Verifying Metrics Robustness ---")
    
    # Setup
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ2(M=3)
    exp.moea = mb.moeas.NSGA3(population=50, generations=10, seed=42)
    exp.run(repeat=2)
    
    metrics = [mb.hv, mb.igd, mb.gd, mb.gdplus, mb.igdplus]
    
    expected_gens = len(exp.runs[0])
    
    for metric_fn in metrics:
        name = metric_fn.__name__
        print(f"\nTesting {name}:")
        
        # 1. Test Experiment
        res_exp = metric_fn(exp)
        print(f"  Experiment: {res_exp}")
        assert res_exp.values.shape == (expected_gens, 2), f"{name} experiment shape mismatch: {res_exp.values.shape} vs {(expected_gens, 2)}"
        
        # 2. Test Run
        res_run = metric_fn(exp.runs[0])
        print(f"  Run:        {res_run}")
        assert res_run.values.shape == (expected_gens, 1), f"{name} run shape mismatch: {res_run.values.shape} vs {(expected_gens, 1)}"
        
        # 3. Test Population
        res_pop = metric_fn(exp.last_pop)
        print(f"  Population: {res_pop}")
        assert res_pop.values.shape == (1, 1), f"{name} population shape mismatch"
        assert isinstance(float(res_pop), float)
        
        # 4. Test Array
        res_arr = metric_fn(exp.front())
        print(f"  Array:      {res_arr}")
        assert res_arr.values.shape == (1, 1), f"{name} array shape mismatch"

    print("\nSUCCESS: All metrics are robust to different input types!")

if __name__ == "__main__":
    verify()
