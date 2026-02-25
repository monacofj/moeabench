import MoeaBench as mb
import numpy as np

mop = mb.mops.DPF2(M=3)
exp = mb.experiment()
exp.mop = mop
exp.moea = mb.moeas.NSGA2(population=120, generations=80)
exp.run(quiet=True)

gt_pts = mop.pf(n_points=2000)
attainment = mb.stats.topo_attainment([gt_pts], name="Optimal Front (GT)")
full_pop = exp.runs[0].pop().objs

print("Rendering test...")
try:
    mb.view.topo_shape(full_pop, attainment, 
                       title="Pathology: Resource Starvation (Collapsed Front)",
                       labels=["Strangled Pop", "Optimal Front (GT)"],
                       markers=True,
                       ref=gt_pts,
                       problem=mop.name,
                       k=120,
                       marker_styles=[None, {'color': 'lightgray', 'size': 1}],
                       show=False) 
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
