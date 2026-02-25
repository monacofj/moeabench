import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.random.rand(500)
y = np.random.rand(500)
z = np.random.rand(500)

try:
    ax.scatter(x, y, z, s=None)
    print("s=None works")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("s=None fails")
