import numpy as np
import pandas as pd
import os
import sys

# Ensure we can import MoeaBench if not installed
sys.path.append(os.path.abspath("."))

from MoeaBench.core import Experiment
from MoeaBench.mops import DTLZ8
from MoeaBench.moeas import NSGA3

def generate_dtlz8_gt(m_list=[10]):
    output_dir = "MoeaBench/mops/data/"
    os.makedirs(output_dir, exist_ok=True)
    
    for M in m_list:
        print(f"\n>>> Generating DTLZ8 Ground Truth for M={M}...")
        
        # 1. Setup Problem (Default N = 10*M)
        mop = DTLZ8(M=M)
        
        # 2. Setup MOEA with High Fidelity
        # Pop=1000 and Gen=2000 is heavy but necessary for the "cliff" of DTLZ8
        # We using pymoo's NSGA-III via our wrapper.
        moea = NSGA3(population=1000, generations=2000, seed=42)
        
        # 3. Setup Experiment
        exp = Experiment(mop=mop, moea=moea)
        exp.name = f"DTLZ8_{M}_GT_Generation"
        
        # 4. Run
        # Constraints are handled automatically by BasePymoo via mop.get_n_ieq_constr()
        print(f"    Running NSGA-III (Pop=1000, Gen=2000)...")
        exp.run()
        
        # 5. Extract Non-Dominated results from the LAST generation
        # We want the absolute best front reached.
        final_pop = exp.last_pop.non_dominated()
        front = final_pop.objectives
        pset = final_pop.variables
        
        # 6. Save to CSV
        f_file = f"{output_dir}DTLZ8_{M}_optimal_f.csv"
        x_file = f"{output_dir}DTLZ8_{M}_optimal_x.csv"
        
        pd.DataFrame(np.array(front)).to_csv(f_file, header=False, index=False)
        pd.DataFrame(np.array(pset)).to_csv(x_file, header=False, index=False)
        
        print(f"    [SUCCESS] Saved M={M}")
        print(f"    Points found: {len(front)}")
        print(f"    Files: {f_file}, {x_file}")

if __name__ == "__main__":
    generate_dtlz8_gt()
