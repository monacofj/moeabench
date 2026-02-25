import MoeaBench as mb
import numpy as np
from MoeaBench.diagnostics.qscore import q_closeness_points

mop = mb.mops.DTLZ1(M=3)
exp = mb.experiment()
exp.mop = mop
exp.moea = mb.moeas.NSGA2(population=500, generations=5)
exp.run(quiet=True)

pop = exp.runs[0].pop()
gt = mop.pf(n_points=1000)

q = q_closeness_points(pop.objs, ref=gt, problem=mop.name, k=500)
print(f"DTLZ1 Gen 5 - Solid: {np.sum(q >= 0.5)}, Hollow: {np.sum((q >= 0.0) & (q < 0.5))}, Diamond: {np.sum(q < 0.0)}")
print(f"Max Obj: {np.max(pop.objs, axis=0)}")
print(f"Min Obj: {np.min(pop.objs, axis=0)}")
