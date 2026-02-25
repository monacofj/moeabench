import MoeaBench as mb
import numpy as np

# Simple mock of the exception
exp = mb.experiment()
exp.mop = mb.mops.DPF2(M=3)
exp.moea = mb.moeas.NSGA2(population=500, generations=30)
exp.run(quiet=True)

gt = exp.mop.pf(n_points=1000)
full_pop = exp.runs[0].pop()

try:
    mb.view.topo_shape(full_pop, gt, 
                       title="Pathology: Resource Starvation (Collapsed Front)",
                       labels=["Strangled Pop", "Optimal Front (GT)"],
                       trace_modes=['markers', 'markers'],
                       marker_styles=[None, {'color': 'lightgray', 'size': 3, 'symbol': 'circle'}],
                       markers=True,
                       show=False) # Headless mode 
    print("SUCCESS")
    # ...
except Exception as e:
    import traceback, sys
    tb = getattr(sys, 'last_traceback', None)
    if tb is None: tb = sys.exc_info()[2]
    # To traverse local vars inside matplotlib, let's catch where it failed if possible
    # Just print the frame variables of the deepest frame we can access from traceback.
    while tb.tb_next:
        tb = tb.tb_next
    print("VARS in failing frame:", list(tb.tb_frame.f_locals.keys()))
    if 's' in tb.tb_frame.f_locals:
        s_val = tb.tb_frame.f_locals['s']
        print("s value and shape:", type(s_val), getattr(s_val, 'shape', None), getattr(s_val, 'size', None))
    if 'x' in tb.tb_frame.f_locals:
        x_val = tb.tb_frame.f_locals['x']
        print("x value and shape:", type(x_val), getattr(x_val, 'shape', None), getattr(x_val, 'size', None))
    print("FAILED")

