import MoeaBench as mb
import numpy as np
from MoeaBench.diagnostics.qscore import q_closeness_points

mop = mb.mops.DTLZ2(M=3)
gt = mop.pf(n_points=1000)
print("DTLZ2 (M=3, Pop=500) Mix Search:")
for g in range(0, 15):
    exp = mb.experiment()
    exp.mop = mop
    exp.moea = mb.moeas.NSGA2(population=400, generations=g)
    exp.run(quiet=True)
    pop = exp.runs[0].pop()
    # Note: k should be pop size for baseline resolution
    q = q_closeness_points(pop, ref=gt, problem=mop.name, k=400)
    s = np.sum(q >= 0.5)
    h = np.sum((q >= 0.0) & (q < 0.5))
    d = np.sum(q < 0.0)
    print(f"Gen {g:2d}: S={s:3d}, H={h:3d}, D={d:3d}")
