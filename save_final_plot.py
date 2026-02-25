import MoeaBench as mb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mop = mb.mops.DTLZ3(M=3)
exp = mb.experiment()
exp.mop = mop
exp.moea = mb.moeas.NSGA2(population=120, generations=100)
exp.run(quiet=True)

gt_pts = mop.pf(n_points=2000)
attainment = mb.stats.topo_attainment([gt_pts])
attainment.name = "Optimal Front (GT)"

full_pop_objs = exp.runs[0].pop().objs

mb.view.topo_shape(full_pop_objs, attainment, 
                   title="Surface Pathology: Resource Starvation (Final Mixed Front)",
                   labels=["Strangled Population", "Optimal Front (GT)"],
                   markers=True,
                   ref=gt_pts,
                   problem=mop.name,
                   k=120,
                   marker_styles=[None, {'color': 'lightgray', 'alpha': 0.1}],
                   show=False) 

plt.savefig("/home/monaco/.gemini/antigravity/brain/b2a192f1-07bd-4129-a999-3939c2e05f8d/final_dtlz3_mix.png", dpi=150)
