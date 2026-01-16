import matplotlib
matplotlib.use('Agg')
from MoeaBench import mb
import numpy as np

# Mock objects or data
data1 = np.random.rand(10, 2)
data2 = np.random.rand(10, 2)

try:
    # This should NOT fail now
    mb.view.spaceplot(data1, data2, labels=["A", "B"], mode='static')
    print("Fix verified: spaceplot accepts 'labels' argument.")
except TypeError as e:
    print(f"Fix failed: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
