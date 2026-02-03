import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from MoeaBench import mb

def extract_stats():
    exp1 = mb.experiment()
    exp1.name = "NSGA3"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA3(population=100, generations=50)
    exp1.run(repeat=1)

    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.SPEA2(population=100, generations=50)
    exp2.run(repeat=1)

    pop1 = exp1.last_run.pop(10)
    pop2 = exp2.last_run.pop(10)

    # Establish global anchor as in strat_caste2
    global_ref = np.vstack([pop1.objectives, pop2.objectives])
    hv1 = float(mb.metrics.hypervolume(pop1.objectives, ref=global_ref))
    hv2 = float(mb.metrics.hypervolume(pop2.objectives, ref=global_ref))
    anchor = max(hv1, hv2)

    # Stratify pop1
    ranks = pop1.stratify()
    mask = (ranks == 1)
    sub_objs = pop1.objectives[mask]
    
    # Calculate individual qualities
    samples = []
    for j in range(len(sub_objs)):
        val = mb.metrics.hypervolume(sub_objs[j:j+1], ref=global_ref)
        samples.append(float(val) / anchor)
    
    q_stats = np.percentile(samples, [0, 25, 50, 75, 100])
    
    print(f"n: {len(sub_objs)}")
    print(f"min: {q_stats[0]:.4f}")
    print(f"Q1: {q_stats[1]:.4f}")
    print(f"q (median): {q_stats[2]:.4f}")
    print(f"Q3: {q_stats[3]:.4f}")
    print(f"max: {q_stats[4]:.4f}")

if __name__ == "__main__":
    extract_stats()
