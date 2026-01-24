
import MoeaBench as mb
import numpy as np

mop = mb.mops.DTLZ1(M=3)
exp = mb.experiment()
exp.mop = mop
opt = exp.optimal(n_points=5)
print("Current DTLZ1 (M=3) Sum of Objectives (Should be 0.5):")
print(np.sum(opt.objectives, axis=1))
