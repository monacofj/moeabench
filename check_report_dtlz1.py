import mb_path
from MoeaBench import mb
import numpy as np

def check():
    # Setup DTLZ1
    mop = mb.mops.DTLZ1(M=3)
    gt = mop.pf()
    
    # 1. Reproduce Report's NSGA3 (N=200, G=200)
    print("\n--- Testing NSGA-III (Report Config: N=200, G=200) ---")
    nsga3 = mb.moeas.NSGA3(population=200, generations=200)
    exp3 = mb.experiment()
    exp3.mop = mop
    exp3.moea = nsga3
    exp3.run(seed=1)
    pop3 = exp3.objectives
    dists3 = mb.diagnostics.qscore.cdist(pop3, gt).min(axis=1)
    
    s_k = 0.02 # Our manual resolution
    
    q3 = mb.diagnostics.qscore.q_closeness_points(pop3, ref=gt, s_k=s_k, problem="DTLZ1")
    
    num_hollow3 = np.sum((q3 >= 0) & (q3 < 0.5))
    num_solid3 = np.sum(q3 >= 0.5)
    num_diamond3 = np.sum(q3 < 0)
    
    print(f"Max dist: {dists3.max():.4f}")
    print(f"Min Q: {q3.min():.4f}")
    print(f"Hollow points (0 <= Q < 0.5): {num_hollow3}")
    print(f"Solid points (Q >= 0.5): {num_solid3}")
    print(f"Diamond points (Q < 0): {num_diamond3}")

    # 2. Reproduce User's NSGA2 (N=100, G=285)
    print("\n--- Testing NSGA-II (User Config: N=100, G=285) ---")
    nsga2 = mb.moeas.NSGA2(population=100, generations=285)
    exp2 = mb.experiment()
    exp2.mop = mop
    exp2.moea = nsga2
    exp2.run(seed=1)
    pop2 = exp2.objectives
    dists2 = mb.diagnostics.qscore.cdist(pop2, gt).min(axis=1)
    
    q2 = mb.diagnostics.qscore.q_closeness_points(pop2, ref=gt, s_k=s_k, problem="DTLZ1")
    
    num_hollow2 = np.sum((q2 >= 0) & (q2 < 0.5))
    num_solid2 = np.sum(q2 >= 0.5)
    num_diamond2 = np.sum(q2 < 0)
    
    print(f"Max dist: {dists2.max():.4f}")
    print(f"Min Q: {q2.min():.4f}")
    print(f"Hollow points (0 <= Q < 0.5): {num_hollow2}")
    print(f"Solid points (Q >= 0.5): {num_solid2}")
    print(f"Diamond points (Q < 0): {num_diamond2}")

if __name__ == "__main__":
    check()
