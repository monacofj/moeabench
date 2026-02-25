import MoeaBench as mb
import numpy as np
from MoeaBench.diagnostics.qscore import q_closeness_points

print("Searching for a mixed front on DPF2...")
try:
    mop = mb.mops.DPF2(M=3)
    for gens in [2, 5, 10, 15, 20]:
        exp = mb.experiment()
        exp.mop = mop
        exp.moea = mb.moeas.NSGA2(population=500, generations=gens)
        exp.run(quiet=True)
        try:
            q_vals = q_closeness_points(exp)
            solid = np.sum(q_vals >= 0.5)
            hollow = np.sum((q_vals >= 0.0) & (q_vals < 0.5))
            diamond = np.sum(q_vals < 0.0)
            print(f"Gen {gens:3d}: Solid: {solid:3d}, Hollow: {hollow:3d}, Diamond: {diamond:3d}")
        except Exception as e:
            pass
except Exception as e:
     print("Error:", e)
