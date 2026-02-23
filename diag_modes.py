import MoeaBench as mb
from MoeaBench.mops import DTLZ2
from MoeaBench.core import Experiment
import logging

logging.basicConfig(level=logging.ERROR)

def test_hv_modes():
    # Use kwags properly for mb.run
    dtlz2 = DTLZ2(n_var=12, n_obj=3)
    exp1 = mb.run(dtlz2, "NSGAII", n_runs=1, n_gen=50, pop_size=10, name="NSGAII_Small")
    exp2 = mb.run(dtlz2, "NSGAII", n_runs=1, n_gen=50, pop_size=100, name="NSGAII_Large")

    print("\n--- Testing Single Experiment (EXP1) ---")
    hv_raw_1 = mb.metrics.hv(exp1, scale='raw')
    hv_rat_1 = mb.metrics.hv(exp1, scale='ratio')
    print(hv_raw_1.report())
    print("\n")
    print(hv_rat_1.report())

    print("\n\n--- Testing Multiple Experiments (EXP1 within EXP1+EXP2) ---")
    res = mb.metrics.hv(exp1, exp2, scale='raw')
    hv_raw_12_1 = res[0]
    res_rat = mb.metrics.hv(exp1, exp2, scale='ratio')
    hv_rat_12_1 = res_rat[0]
    
    print(hv_raw_12_1.report())
    print("\n")
    print(hv_rat_12_1.report())

if __name__ == "__main__":
    test_hv_modes()
