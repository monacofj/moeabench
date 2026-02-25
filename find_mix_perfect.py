import MoeaBench as mb
import numpy as np
from MoeaBench.diagnostics.qscore import q_closeness_points

def scan(mop_name, mop_obj):
    print(f"\nScanning {mop_name} (Pop=500):")
    gt = mop_obj.pf(n_points=1000)
    for g in [0, 2, 5, 10, 20]:
        exp = mb.experiment()
        exp.mop = mop_obj
        exp.moea = mb.moeas.NSGA2(population=400, generations=g)
        exp.run(quiet=True)
        pop = exp.runs[0].pop()
        full_objs = pop.objs
        q = q_closeness_points(full_objs, ref=gt, problem=mop_obj.name, k=400)
        s = np.sum(q >= 0.5)
        h = np.sum((q >= 0.0) & (q < 0.5))
        d = np.sum(q < 0.0)
        print(f"Gen {g:2d}: S={s:3d}, H={h:3d}, D={d:3d} (Total={len(q)})")

scan("DPF2", mb.mops.DPF2(M=3))
scan("DTLZ2", mb.mops.DTLZ2(M=3))
