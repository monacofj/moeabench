import MoeaBench as mb
import numpy as np
from MoeaBench.diagnostics.qscore import q_closeness_points

print("Searching for a mixed front on full population...")
# Test DTLZ1
mop_dtlz1 = mb.mops.DTLZ1(M=3)
for gens in [2, 5, 10, 20]:
    exp = mb.experiment()
    exp.mop = mop_dtlz1
    exp.moea = mb.moeas.NSGA2(population=500, generations=gens)
    exp.run(quiet=True)
    try:
        full_pop = exp.runs[0].pop()
        gt = exp.mop.pf(n_points=1000)
        q_vals = q_closeness_points(full_pop, ref=gt, problem=exp.mop.name, k=len(full_pop))
        solid = np.sum(q_vals >= 0.5)
        hollow = np.sum((q_vals >= 0.0) & (q_vals < 0.5))
        diamond = np.sum(q_vals < 0.0)
        print(f"DTLZ1 Gen {gens:3d}: Solid: {solid:3d}, Hollow: {hollow:3d}, Diamond: {diamond:3d}")
    except Exception as e:
        pass

# Test DTLZ3
mop_dtlz3 = mb.mops.DTLZ3(M=3)
for gens in [2, 5, 10, 20, 30]:
    exp = mb.experiment()
    exp.mop = mop_dtlz3
    exp.moea = mb.moeas.NSGA2(population=500, generations=gens)
    exp.run(quiet=True)
    try:
        full_pop = exp.runs[0].pop()
        gt = exp.mop.pf(n_points=1000)
        q_vals = q_closeness_points(full_pop, ref=gt, problem=exp.mop.name, k=len(full_pop))
        solid = np.sum(q_vals >= 0.5)
        hollow = np.sum((q_vals >= 0.0) & (q_vals < 0.5))
        diamond = np.sum(q_vals < 0.0)
        print(f"DTLZ3 Gen {gens:3d}: Solid: {solid:3d}, Hollow: {hollow:3d}, Diamond: {diamond:3d}")
    except Exception as e:
        pass
