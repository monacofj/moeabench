import MoeaBench as mb
import numpy as np
import os

# 1. Setup a test scenario
print("Testing Visual Quality Markers...")
from MoeaBench.mops.DTLZ2 import DTLZ2
mop = DTLZ2(M=3)
pf = mop.pf()

# 2. Create a "Mixed Quality" population
# Points 0-10: Perfect (from PF) -> Circles
# Points 10-20: Slightly noisy -> Squares
# Points 20-30: Very noisy -> Diamonds
P = pf[:30].copy()
P[10:20] += 0.05 * np.random.rand(10, 3)
P[20:30] += 0.5 * np.random.rand(10, 3)

# 3. Plot with markers=True
# We need to mock a bit because topo_shape expects an object with metadata if possible
class DummyRun:
    def __init__(self, objectives, mop):
        self.objectives = objectives
        self.mop = mop
        self.name = "Mixed Quality Run"

run = DummyRun(P, mop)

# Generate plot (save to file for verification)
print("Generating 3D Plot...")
s = mb.view.topo_shape(run, markers=True, show=False)
s.figure.write_html("test_visual_markers.html")
print("Plot saved to test_visual_markers.html")

# 4. Verify 2D
print("Generating 2D Plot...")
s2 = mb.view.topo_shape(run, objectives=[0, 1], markers=True, show=False)
s2.figure.write_html("test_visual_markers_2d.html")
print("Plot saved to test_visual_markers_2d.html")
