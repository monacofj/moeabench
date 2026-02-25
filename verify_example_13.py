import MoeaBench as mb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mop = mb.mops.DPF2(M=3)
exp = mb.experiment()
exp.mop = mop
exp.moea = mb.moeas.NSGA2(population=120, generations=80)
exp.run(quiet=True)

gt_pts = mop.pf(n_points=2000)
attainment = mb.stats.topo_attainment([gt_pts])
attainment.name = "Optimal Front (GT)"

full_pop_objs = exp.runs[0].pop().objs

print("Calculating Q-Scores for verification...")
from MoeaBench.diagnostics.qscore import q_closeness_points
q = q_closeness_points(full_pop_objs, ref=gt_pts, problem=mop.name, k=120)
print(f"Mix: S={np.sum(q>=0.5)}, H={np.sum((q>=0.0)&(q<0.5))}, D={np.sum(q<0.0)}")

print("Generating Plot...")
mb.view.topo_shape(full_pop_objs, attainment, 
                   title="Surface Pathology: Resource Starvation (Mixed Quality Front)",
                   labels=["Strangled Population", "Optimal Front (GT)"],
                   markers=True,
                   ref=gt_pts,
                   problem=mop.name,
                   k=120,
                   marker_styles=[None, {'color': 'lightgray', 'alpha': 0.1}],
                   show=False) 

plt.savefig("example_13_verification.png", dpi=150)
print("Saved to example_13_verification.png")
