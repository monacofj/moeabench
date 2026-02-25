import MoeaBench as mb
import numpy as np
from MoeaBench.diagnostics.qscore import q_closeness_points

exp = mb.experiment()
exp.mop = mb.mops.DPF2(M=3)
exp.moea = mb.moeas.NSGA2(population=500, generations=30)
exp.run(quiet=True)

gt = exp.mop.pf(n_points=1000)
full_pop = exp.runs[0].pop()

# How q_closeness_points behaves on full_pop
q_vals = q_closeness_points(full_pop, ref=gt, problem=exp.mop.name, k=len(full_pop))
print("full_pop Q-scores:", len(q_vals))
print("Solid (>=0.5):", np.sum(q_vals >= 0.5))
print("Hollow (>=0.0):", np.sum((q_vals >= 0.0) & (q_vals < 0.5)))
print("Diamond (<0.0):", np.sum(q_vals < 0.0))

# What happens to GT?
gt_q_vals = q_closeness_points(gt, ref=gt, problem=exp.mop.name, k=len(full_pop))
print("GT Q-scores:", len(gt_q_vals))
