import MoeaBench as mb
import numpy as np
from MoeaBench.diagnostics.qscore import q_closeness_points

mop = mb.mops.DPF2(M=3)
gt = mop.pf(n_points=2000)
print("DPF2 Fine Grained Search:")
for g in [2, 3, 4, 5]:
    exp = mb.experiment()
    exp.mop = mop
    exp.moea = mb.moeas.NSGA2(population=500, generations=g)
    exp.run(quiet=True)
    pop = exp.runs[0].pop()
    full_objs = pop.objs
    q = q_closeness_points(full_objs, ref=gt, problem=mop.name, k=500)
    s = np.sum(q >= 0.5)
    h = np.sum((q >= 0.0) & (q < 0.5))
    d = np.sum(q < 0.0)
    print(f"Gen {g:d}: S={s:3d}, H={h:3d}, D={d:3d}")
