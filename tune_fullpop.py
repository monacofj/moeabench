import MoeaBench as mb
import numpy as np
from MoeaBench.diagnostics.qscore import q_closeness_points

print("Searching for a mixed front on full population...")
mop = mb.mops.DTLZ2(M=3)
for gens in [2, 10, 20, 30, 50, 80]:
    exp = mb.experiment()
    exp.mop = mop
    exp.name = f"DTLZ2_pop_500_{gens}"
    exp.moea = mb.moeas.NSGA2(population=500, generations=gens)
    exp.run(quiet=True)
    try:
        full_pop = exp.runs[0].pop()
        gt = exp.mop.pf(n_points=1000)
        q_vals = q_closeness_points(full_pop, ref=gt, problem=exp.mop.name, k=len(full_pop))
        solid = np.sum(q_vals >= 0.5)
        hollow = np.sum((q_vals >= 0.0) & (q_vals < 0.5))
        diamond = np.sum(q_vals < 0.0)
        print(f"Gen {gens:3d}: Solid: {solid:3d}, Hollow: {hollow:3d}, Diamond: {diamond:3d}")
    except Exception as e:
        print("Error:", e)
